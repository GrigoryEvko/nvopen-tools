// Function: sub_29A6EE0
// Address: 0x29a6ee0
//
__int64 __fastcall sub_29A6EE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r13
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdi
  _BYTE v10[49]; // [rsp+Fh] [rbp-31h] BYREF

  v3 = a1 + 32;
  v4 = a3 + 40;
  v6 = *(_QWORD *)(a3 + 48);
  v10[0] = 0;
  v7 = a1 + 80;
  if ( v4 == v6 )
    goto LABEL_8;
  do
  {
    v8 = v6 - 48;
    if ( !v6 )
      v8 = 0;
    sub_29A6D90(v8, v10);
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v4 != v6 );
  v3 = a1 + 32;
  v7 = a1 + 80;
  if ( !v10[0] )
  {
LABEL_8:
    *(_QWORD *)(a1 + 8) = v3;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v7;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    memset((void *)a1, 0, 0x60u);
    *(_QWORD *)(a1 + 8) = v3;
    *(_DWORD *)(a1 + 16) = 2;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 56) = v7;
    *(_DWORD *)(a1 + 64) = 2;
    *(_BYTE *)(a1 + 76) = 1;
  }
  return a1;
}
