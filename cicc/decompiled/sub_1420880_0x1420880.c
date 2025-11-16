// Function: sub_1420880
// Address: 0x1420880
//
void __fastcall sub_1420880(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  bool v5; // zf
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // rax
  bool v9; // cc
  __int64 v10; // rax
  __int64 v11; // r9
  char v12; // al
  _QWORD *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdx
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  _QWORD v20[4]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD *v21; // [rsp+20h] [rbp-40h]
  __int64 v22; // [rsp+28h] [rbp-38h]
  _QWORD v23[6]; // [rsp+30h] [rbp-30h] BYREF

  v2 = *(_QWORD *)(a1 + 48);
  if ( *(_BYTE *)(v2 + 16) == 23 )
  {
    if ( (*(_BYTE *)(v2 + 23) & 0x40) != 0 )
      v3 = *(_QWORD *)(v2 - 8);
    else
      v3 = v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF);
    v4 = *(_QWORD *)(v3 + 24LL * *(unsigned int *)(a1 + 56));
  }
  else
  {
    v4 = *(_QWORD *)(v2 - 24);
  }
  v5 = *(_BYTE *)(a1 + 112) == 0;
  *(_QWORD *)a1 = v4;
  if ( v5 || !*(_QWORD *)(a1 + 64) )
    goto LABEL_19;
  v6 = sub_157EB90(*(_QWORD *)(*(_QWORD *)(a1 + 104) + 64LL));
  v20[2] = 0;
  v7 = sub_1632FA0(v6);
  v8 = *(_QWORD *)(a1 + 64);
  v20[3] = 0;
  v21 = v23;
  v22 = 0x400000000LL;
  v9 = *(_BYTE *)(v8 + 16) <= 0x17u;
  v20[0] = v8;
  v20[1] = v7;
  if ( !v9 )
  {
    v23[0] = v8;
    LODWORD(v22) = 1;
  }
  v10 = *(_QWORD *)(a1 + 48);
  if ( *(_BYTE *)(v10 + 16) != 23 )
    BUG();
  if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
    v11 = *(_QWORD *)(v10 - 8);
  else
    v11 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
  v12 = sub_143C480(
          v20,
          *(_QWORD *)(*(_QWORD *)(a1 + 104) + 64LL),
          *(_QWORD *)(v11 + 8LL * *(unsigned int *)(a1 + 56) + 8 + 24LL * *(unsigned int *)(v10 + 76)),
          0,
          0);
  v13 = v21;
  if ( v12 || *(_QWORD *)(a1 + 64) == v20[0] )
  {
    if ( v21 != v23 )
      _libc_free((unsigned __int64)v21);
LABEL_19:
    v18 = _mm_loadu_si128((const __m128i *)(a1 + 64));
    v19 = _mm_loadu_si128((const __m128i *)(a1 + 80));
    *(_QWORD *)(a1 + 40) = *(_QWORD *)(a1 + 96);
    *(__m128i *)(a1 + 8) = v18;
    *(__m128i *)(a1 + 24) = v19;
    return;
  }
  v14 = *(_QWORD *)(a1 + 72);
  v15 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 8) = v20[0];
  v16 = *(_QWORD *)(a1 + 88);
  v17 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 16) = v14;
  *(_QWORD *)(a1 + 24) = v15;
  *(_QWORD *)(a1 + 32) = v16;
  *(_QWORD *)(a1 + 40) = v17;
  if ( v13 != v23 )
    _libc_free((unsigned __int64)v13);
}
