// Function: sub_258EBC0
// Address: 0x258ebc0
//
char __fastcall sub_258EBC0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r13
  unsigned int v9; // eax
  unsigned int v10; // edx
  unsigned __int64 v11; // r8
  __m128i *v12; // rax
  __m128i *v13; // rdx
  unsigned __int64 v14; // rax
  __int128 *v15; // rdx
  __int64 v16; // rax
  _DWORD *v17; // rdi
  __int64 (*v18)(void); // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  int v22; // edi
  unsigned __int64 v23; // rsi
  char v24; // dl
  __int64 v25; // [rsp-10h] [rbp-B0h]
  __int128 v26; // [rsp+20h] [rbp-80h] BYREF
  __int128 v27; // [rsp+30h] [rbp-70h] BYREF
  __m128i *v28; // [rsp+40h] [rbp-60h]
  __int128 *v29; // [rsp+48h] [rbp-58h]
  __int64 v30; // [rsp+50h] [rbp-50h]
  void *v31; // [rsp+58h] [rbp-48h]
  __int64 v32; // [rsp+60h] [rbp-40h]

  v3 = sub_250D2C0(a2, **(_QWORD **)a1);
  v5 = sub_258DCE0(*(_QWORD *)(a1 + 8), v3, v4, *(_QWORD *)(a1 + 16), 0, 0, 1);
  if ( !v5 )
    return 0;
  v6 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v5 + 48LL))(v5, v3, v25);
  v7 = *(_QWORD *)(a1 + 24);
  v8 = v6;
  if ( !*(_BYTE *)(v7 + 88) )
  {
    v26 = 0;
    v32 = 256;
    v27 = 0;
    v30 = 0;
    v28 = (__m128i *)&v27;
    v29 = &v27;
    v31 = &unk_4A16CD8;
    *(_QWORD *)v7 = &unk_4A16DD8;
    *(_DWORD *)(v7 + 16) = v26;
    *(_QWORD *)(v7 + 8) = &unk_4A16D78;
    *(_DWORD *)(v7 + 20) = -1;
    v20 = *((_QWORD *)&v27 + 1);
    v21 = v7 + 32;
    if ( *((_QWORD *)&v27 + 1) )
    {
      v22 = v27;
      *(_QWORD *)(v7 + 40) = *((_QWORD *)&v27 + 1);
      *(_DWORD *)(v7 + 32) = v22;
      *(_QWORD *)(v7 + 48) = v28;
      *(_QWORD *)(v7 + 56) = v29;
      *(_QWORD *)(v20 + 8) = v21;
      v23 = 0;
      *(_QWORD *)(v7 + 64) = v30;
      *((_QWORD *)&v27 + 1) = 0;
      v28 = (__m128i *)&v27;
      v29 = &v27;
      v30 = 0;
    }
    else
    {
      *(_DWORD *)(v7 + 32) = 0;
      *(_QWORD *)(v7 + 40) = 0;
      *(_QWORD *)(v7 + 48) = v21;
      *(_QWORD *)(v7 + 56) = v21;
      *(_QWORD *)(v7 + 64) = 0;
      v23 = *((_QWORD *)&v27 + 1);
    }
    *(_BYTE *)(v7 + 80) = v32;
    v24 = BYTE1(v32);
    *(_QWORD *)(v7 + 72) = &unk_4A16CD8;
    *(_BYTE *)(v7 + 81) = v24;
    *(_BYTE *)(v7 + 88) = 1;
    sub_255C230((__int64)&v26 + 8, v23);
    v7 = *(_QWORD *)(a1 + 24);
  }
  v9 = *(_DWORD *)(v8 + 20);
  v10 = *(_DWORD *)(v7 + 16);
  if ( *(_DWORD *)(v8 + 16) <= v10 )
    v10 = *(_DWORD *)(v8 + 16);
  if ( *(_DWORD *)(v7 + 20) <= v9 )
    v9 = *(_DWORD *)(v7 + 20);
  *(_DWORD *)(v7 + 16) = v10;
  *(_DWORD *)(v7 + 20) = v9;
  *(_WORD *)(v7 + 80) &= *(_WORD *)(v8 + 80);
  LODWORD(v27) = 0;
  *((_QWORD *)&v27 + 1) = 0;
  v28 = (__m128i *)&v27;
  v29 = &v27;
  v30 = 0;
  v11 = *(_QWORD *)(v7 + 40);
  if ( v11 )
  {
    v12 = sub_25394D0(*(const __m128i **)(v7 + 40), (__int64)&v27);
    v11 = (unsigned __int64)v12;
    do
    {
      v13 = v12;
      v12 = (__m128i *)v12[1].m128i_i64[0];
    }
    while ( v12 );
    v28 = v13;
    v14 = v11;
    do
    {
      v15 = (__int128 *)v14;
      v14 = *(_QWORD *)(v14 + 24);
    }
    while ( v14 );
    v29 = v15;
    v16 = *(_QWORD *)(v7 + 64);
    *((_QWORD *)&v27 + 1) = v11;
    v30 = v16;
  }
  sub_255C230((__int64)&v26 + 8, v11);
  v17 = *(_DWORD **)(a1 + 24);
  v18 = *(__int64 (**)(void))(*(_QWORD *)v17 + 16LL);
  if ( (char *)v18 == (char *)sub_2505DB0 )
    return v17[5] != 0;
  else
    return v18();
}
