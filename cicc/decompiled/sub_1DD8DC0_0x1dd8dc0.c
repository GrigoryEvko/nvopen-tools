// Function: sub_1DD8DC0
// Address: 0x1dd8dc0
//
char *__fastcall sub_1DD8DC0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  _BYTE *v4; // rsi
  _BYTE *v5; // rsi
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 i; // rbx
  char *result; // rax
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD **)(a2 + 56);
  v10[0] = a2;
  v4 = (_BYTE *)v3[13];
  if ( v4 == (_BYTE *)v3[14] )
  {
    result = sub_1D4AF10((__int64)(v3 + 12), v4, v10);
    v5 = (_BYTE *)v3[13];
  }
  else
  {
    if ( v4 )
    {
      *(_QWORD *)v4 = a2;
      v4 = (_BYTE *)v3[13];
    }
    v5 = v4 + 8;
    v3[13] = v5;
  }
  v6 = a2 + 24;
  *(_DWORD *)(v6 + 24) = ((__int64)&v5[-v3[12]] >> 3) - 1;
  v7 = v3[5];
  for ( i = *(_QWORD *)(v6 + 8); v6 != i; i = *(_QWORD *)(i + 8) )
    result = (char *)sub_1E15C30(i, v7);
  return result;
}
