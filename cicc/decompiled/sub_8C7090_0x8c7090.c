// Function: sub_8C7090
// Address: 0x8c7090
//
_QWORD *__fastcall sub_8C7090(char a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 *v3; // r13
  __int64 v4; // rax
  _QWORD *result; // rax
  __int64 v6; // rdx

  if ( a1 == 37 )
  {
    v2 = *(_QWORD **)(a2 + 64);
    v3 = (__int64 *)(a2 + 64);
  }
  else
  {
    v2 = *(_QWORD **)(a2 + 32);
    v3 = (__int64 *)(a2 + 32);
  }
  if ( v2 )
  {
    if ( a2 != *v2 )
    {
LABEL_5:
      sub_8D0810(*v3);
      *v3 = 0;
LABEL_6:
      v4 = sub_8D07C0();
      *v3 = v4;
      *(_BYTE *)(v4 + 20) = a1;
      ++*(_DWORD *)(*v3 + 16);
      result = (_QWORD *)*v3;
      goto LABEL_7;
    }
    v6 = v2[1];
    if ( a2 != v6 && v6 )
    {
      *v2 = v6;
      goto LABEL_5;
    }
  }
  result = (_QWORD *)*v3;
  if ( !*v3 )
    goto LABEL_6;
LABEL_7:
  *result = a2;
  if ( (*(_BYTE *)(a2 - 8) & 2) == 0 )
  {
    result = (_QWORD *)*v3;
    *(_QWORD *)(*v3 + 8) = a2;
  }
  return result;
}
