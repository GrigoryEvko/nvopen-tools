// Function: sub_823310
// Address: 0x823310
//
_QWORD *__fastcall sub_823310(int a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // r15
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  int v7; // r12d
  _QWORD *v8; // rdi
  __int64 v9; // rax
  _QWORD *result; // rax

  if ( !a1 )
  {
    v3 = qword_4F195D0;
    if ( qword_4F195D0 )
    {
      sub_822B90(*(_QWORD *)qword_4F195D0, 16LL * (unsigned int)(*(_DWORD *)(qword_4F195D0 + 8) + 1));
      a2 = 16;
      sub_822B90(v3, 16);
      qword_4F195D0 = 0;
    }
  }
  v4 = 8LL * a1;
  v5 = (char *)qword_4F073B0 + v4;
  v6 = *(_QWORD **)((char *)qword_4F073B0 + v4);
  if ( v6 )
  {
    v7 = dword_4F195D8;
    do
    {
      v8 = v6;
      v6 = (_QWORD *)*v6;
      if ( v7 && (v9 = v8[4]) != 0 && v9 == v8[3] - (_QWORD)v8 )
      {
        _libc_free(v8, a2);
      }
      else
      {
        *v8 = qword_4F195E0;
        qword_4F195E0 = (__int64)v8;
      }
    }
    while ( v6 );
    v5 = (char *)qword_4F073B0 + v4;
  }
  *v5 = 0;
  *((_QWORD *)qword_4F072B0 + a1) = 0;
  result = dword_4F073B8;
  if ( dword_4F073B8[0] == a1 )
  {
    result = &qword_4F06BB8;
    qword_4F06BB8 = 0;
  }
  return result;
}
