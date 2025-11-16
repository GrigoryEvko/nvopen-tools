// Function: sub_DC1960
// Address: 0xdc1960
//
_QWORD *__fastcall sub_DC1960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  bool v6; // zf
  unsigned int v7; // ecx
  _QWORD *v8; // r12
  char v10; // [rsp+Ch] [rbp-54h]
  __int64 *v11; // [rsp+10h] [rbp-50h] BYREF
  __int64 v12; // [rsp+18h] [rbp-48h]
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  char v14[56]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *(_WORD *)(a3 + 24) == 8;
  v11 = &v13;
  v13 = a2;
  v12 = 0x400000001LL;
  if ( v6 && a4 == *(_QWORD *)(a3 + 48) )
  {
    v10 = a5;
    sub_D932D0((__int64)&v11, v14, *(char **)(a3 + 32), (char *)(*(_QWORD *)(a3 + 32) + 8LL * *(_QWORD *)(a3 + 40)));
    v7 = v10 & 1;
  }
  else
  {
    *(_QWORD *)v14 = a3;
    v7 = a5;
    LODWORD(v12) = 2;
  }
  v8 = sub_DBFF60(a1, (unsigned int *)&v11, a4, v7);
  if ( v11 != &v13 )
    _libc_free(v11, &v11);
  return v8;
}
