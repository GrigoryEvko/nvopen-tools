// Function: sub_980680
// Address: 0x980680
//
unsigned int __fastcall sub_980680(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  __int64 v11; // rax
  __m128i v13[9]; // [rsp+0h] [rbp-100h] BYREF
  __int64 v14; // [rsp+90h] [rbp-70h]
  unsigned int v15; // [rsp+A0h] [rbp-60h]
  __int64 v16; // [rsp+B0h] [rbp-50h]
  __int64 v17; // [rsp+C0h] [rbp-40h]
  __int64 v18; // [rsp+C8h] [rbp-38h]
  __int64 v19; // [rsp+D8h] [rbp-28h]

  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 4;
  *(_QWORD *)(a1 + 16) = &unk_4F6D3F0;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)a1 = &unk_49D9670;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  sub_980550(v13, a2);
  sub_97F4E0(a1 + 176, (__int64)v13);
  v3 = v18;
  *(_BYTE *)(a1 + 400) = 1;
  if ( v3 )
    j_j___libc_free_0(v3, v19 - v3);
  if ( v16 )
    j_j___libc_free_0(v16, v17 - v16);
  v4 = v15;
  if ( v15 )
  {
    v5 = v14;
    v6 = v14 + 40LL * v15;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v5 <= 0xFFFFFFFD )
        {
          v7 = *(_QWORD *)(v5 + 8);
          if ( v7 != v5 + 24 )
            break;
        }
        v5 += 40;
        if ( v6 == v5 )
          goto LABEL_11;
      }
      v8 = *(_QWORD *)(v5 + 24);
      v5 += 40;
      j_j___libc_free_0(v7, v8 + 1);
    }
    while ( v6 != v5 );
LABEL_11:
    v4 = v15;
  }
  v9 = v14;
  v10 = 40 * v4;
  sub_C7D6A0(v14, 40 * v4, 8);
  *(_BYTE *)(a1 + 488) = 0;
  v11 = sub_BC2B00(v9, v10);
  return sub_97FFF0(v11);
}
