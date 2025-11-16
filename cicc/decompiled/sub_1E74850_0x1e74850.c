// Function: sub_1E74850
// Address: 0x1e74850
//
void __fastcall sub_1E74850(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 *i; // r15
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-60h] BYREF
  int v12; // [rsp+8h] [rbp-58h]
  __int64 v13; // [rsp+10h] [rbp-50h]
  __int128 v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+28h] [rbp-38h]

  v2 = (__int64 *)a1[25];
  for ( i = (__int64 *)a1[26]; i != v2; ++v2 )
  {
    v5 = *(_QWORD *)a2;
    v6 = *v2;
    v14 = 0u;
    v7 = a1[16];
    v11 = v5;
    LODWORD(v5) = *(_DWORD *)(a2 + 8);
    v13 = v6;
    v12 = v5;
    v8 = a1[2];
    LOWORD(v14) = 256;
    v15 = 0;
    sub_1E736C0((__int64)&v11, v7, v8);
    sub_1E74710((__int64)a1, a2, (__int64)&v11);
    if ( (_BYTE)v14 )
    {
      *(_BYTE *)(a2 + 24) = v14;
      v9 = v13;
      *(_BYTE *)(a2 + 25) = BYTE1(v14);
      v10 = *(_QWORD *)((char *)&v14 + 2);
      *(_QWORD *)(a2 + 16) = v9;
      *(_QWORD *)(a2 + 26) = v10;
      *(_DWORD *)(a2 + 34) = *(_DWORD *)((char *)&v14 + 10);
      *(_QWORD *)(a2 + 40) = v15;
    }
  }
}
