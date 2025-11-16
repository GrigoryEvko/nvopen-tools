// Function: sub_8C9FB0
// Address: 0x8c9fb0
//
_DWORD *__fastcall sub_8C9FB0(__int64 a1, unsigned int a2)
{
  __int64 i; // rbx
  __int64 v3; // r13
  _DWORD *result; // rax
  _QWORD *v5; // rdi
  _QWORD *v6; // rbx
  __int64 v7; // r13

  for ( i = *(_QWORD *)(a1 + 160); i; i = *(_QWORD *)(i + 112) )
  {
    while ( a2 )
    {
      sub_8C7090(8, i);
LABEL_4:
      i = *(_QWORD *)(i + 112);
      if ( !i )
        goto LABEL_8;
    }
    v3 = *(_QWORD *)(i + 32);
    if ( !v3 )
      goto LABEL_4;
    sub_8C6400(8, i);
    sub_8D0810(v3);
    *(_QWORD *)(i + 32) = 0;
  }
LABEL_8:
  result = &dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    result = *(_DWORD **)(a1 + 168);
    v5 = (_QWORD *)*((_QWORD *)result + 19);
    v6 = *(_QWORD **)result;
    if ( v5 )
      result = (_DWORD *)sub_8C9D40(v5, a2);
    while ( v6 )
    {
      if ( a2 )
      {
        result = sub_8C7090(37, (__int64)v6);
      }
      else
      {
        v7 = v6[4];
        if ( v7 )
        {
          sub_8C6400(37, (__int64)v6);
          result = (_DWORD *)sub_8D0810(v7);
          v6[4] = 0;
        }
      }
      v6 = (_QWORD *)*v6;
    }
  }
  return result;
}
