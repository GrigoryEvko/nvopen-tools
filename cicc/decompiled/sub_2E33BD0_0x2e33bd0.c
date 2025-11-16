// Function: sub_2E33BD0
// Address: 0x2e33bd0
//
void __fastcall sub_2E33BD0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r12
  _BYTE *v4; // rsi
  _BYTE *v5; // rsi
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 i; // rbx
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v3 = *(_QWORD **)(a2 + 32);
  v9[0] = a2;
  v4 = (_BYTE *)v3[13];
  if ( v4 == (_BYTE *)v3[14] )
  {
    sub_2E33A40((__int64)(v3 + 12), v4, v9);
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
  *(_DWORD *)(a2 + 24) = ((__int64)&v5[-v3[12]] >> 3) - 1;
  v6 = v3[4];
  v7 = a2 + 48;
  for ( i = *(_QWORD *)(a2 + 56); v7 != i; i = *(_QWORD *)(i + 8) )
    sub_2E86750(i, v6);
}
