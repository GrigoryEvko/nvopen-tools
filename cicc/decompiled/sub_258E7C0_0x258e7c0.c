// Function: sub_258E7C0
// Address: 0x258e7c0
//
__int64 __fastcall sub_258E7C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *v7; // rax
  int v8; // edi
  char *v9; // r12
  signed int v10; // r14d
  __int64 v11; // rdx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // r15
  unsigned int v18; // eax
  unsigned int v19; // edx
  int v20; // eax
  unsigned __int64 v21; // r8
  __m128i *v22; // rax
  __m128i *v23; // rdx
  unsigned __int64 v24; // rax
  __int128 *v25; // rdx
  __int64 v26; // rax
  _DWORD *v27; // rdi
  __int64 (*v28)(void); // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  int v33; // esi
  __int64 v34; // rdx
  __int64 v35; // rsi
  int v36; // edi
  unsigned __int64 v37; // rsi
  char v38; // dl
  __int64 v39; // [rsp-10h] [rbp-F0h]
  __int64 v40; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v41; // [rsp+28h] [rbp-B8h]
  void *v42; // [rsp+30h] [rbp-B0h]
  char *v43; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v44; // [rsp+40h] [rbp-A0h]
  char v45; // [rsp+48h] [rbp-98h] BYREF
  void *v46; // [rsp+50h] [rbp-90h] BYREF
  _OWORD *v47; // [rsp+58h] [rbp-88h] BYREF
  __int128 v48; // [rsp+60h] [rbp-80h] BYREF
  __int128 v49; // [rsp+70h] [rbp-70h] BYREF
  __m128i *v50; // [rsp+80h] [rbp-60h]
  __int128 *v51; // [rsp+88h] [rbp-58h]
  __int64 v52; // [rsp+90h] [rbp-50h]
  void *v53; // [rsp+98h] [rbp-48h]
  __int64 v54; // [rsp+A0h] [rbp-40h]

  v7 = *(void **)a2;
  v8 = *(_DWORD *)(a2 + 16);
  v43 = &v45;
  v44 = 0;
  v42 = v7;
  if ( v8 )
  {
    v9 = (char *)&v48 + 8;
    sub_2538240((__int64)&v43, (char **)(a2 + 8), a3, a4, a5, a6);
    v10 = **(_DWORD **)a1;
    v47 = (__int128 *)((char *)&v48 + 8);
    *(_QWORD *)&v48 = 0;
    v46 = v42;
    if ( (_DWORD)v44 )
      sub_2538550((__int64)&v47, (__int64)&v43, v29, v30, v31, v32);
  }
  else
  {
    v9 = (char *)&v48 + 8;
    v10 = **(_DWORD **)a1;
    v46 = v7;
    v47 = (__int128 *)((char *)&v48 + 8);
    *(_QWORD *)&v48 = 0;
  }
  v40 = sub_254CA10((__int64)&v46, v10);
  v41 = v11;
  if ( v47 != (__int128 *)((char *)&v48 + 8) )
    _libc_free((unsigned __int64)v47);
  if ( (unsigned __int8)sub_2509800(&v40)
    && (v13 = v40, (v14 = sub_258DCE0(*(_QWORD *)(a1 + 8), v40, v41, *(_QWORD *)(a1 + 16), 0, 0, 1)) != 0) )
  {
    v15 = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v14 + 48LL))(v14, v13, v39);
    v16 = *(_QWORD *)(a1 + 24);
    v17 = v15;
    if ( !*(_BYTE *)(v16 + 88) )
    {
      v48 = 0;
      v54 = 256;
      DWORD1(v48) = -1;
      v47 = &unk_4A16D78;
      v49 = 0;
      v52 = 0;
      v50 = (__m128i *)&v49;
      v51 = &v49;
      v53 = &unk_4A16CD8;
      *(_QWORD *)v16 = &unk_4A16DD8;
      *(_DWORD *)(v16 + 16) = v48;
      v33 = DWORD1(v48);
      *(_QWORD *)(v16 + 8) = &unk_4A16D78;
      *(_DWORD *)(v16 + 20) = v33;
      v34 = *((_QWORD *)&v49 + 1);
      v35 = v16 + 32;
      if ( *((_QWORD *)&v49 + 1) )
      {
        v36 = v49;
        *(_QWORD *)(v16 + 40) = *((_QWORD *)&v49 + 1);
        *(_DWORD *)(v16 + 32) = v36;
        *(_QWORD *)(v16 + 48) = v50;
        *(_QWORD *)(v16 + 56) = v51;
        *(_QWORD *)(v34 + 8) = v35;
        v37 = 0;
        *(_QWORD *)(v16 + 64) = v52;
        *((_QWORD *)&v49 + 1) = 0;
        v50 = (__m128i *)&v49;
        v51 = &v49;
        v52 = 0;
      }
      else
      {
        *(_DWORD *)(v16 + 32) = 0;
        *(_QWORD *)(v16 + 40) = 0;
        *(_QWORD *)(v16 + 48) = v35;
        *(_QWORD *)(v16 + 56) = v35;
        *(_QWORD *)(v16 + 64) = 0;
        v37 = *((_QWORD *)&v49 + 1);
      }
      *(_BYTE *)(v16 + 80) = v54;
      v38 = BYTE1(v54);
      *(_QWORD *)(v16 + 72) = &unk_4A16CD8;
      *(_BYTE *)(v16 + 81) = v38;
      *(_BYTE *)(v16 + 88) = 1;
      v46 = &unk_4A16DD8;
      sub_255C230((__int64)&v48 + 8, v37);
      v16 = *(_QWORD *)(a1 + 24);
    }
    v18 = *(_DWORD *)(v17 + 20);
    v19 = *(_DWORD *)(v16 + 16);
    if ( *(_DWORD *)(v17 + 16) <= v19 )
      v19 = *(_DWORD *)(v17 + 16);
    if ( *(_DWORD *)(v16 + 20) <= v18 )
      v18 = *(_DWORD *)(v16 + 20);
    *(_DWORD *)(v16 + 16) = v19;
    *(_DWORD *)(v16 + 20) = v18;
    *(_WORD *)(v16 + 80) &= *(_WORD *)(v17 + 80);
    v46 = &unk_4A16DD8;
    LODWORD(v48) = *(_DWORD *)(v16 + 16);
    v20 = *(_DWORD *)(v16 + 20);
    LODWORD(v49) = 0;
    DWORD1(v48) = v20;
    v47 = &unk_4A16D78;
    *((_QWORD *)&v49 + 1) = 0;
    v50 = (__m128i *)&v49;
    v51 = &v49;
    v52 = 0;
    v21 = *(_QWORD *)(v16 + 40);
    if ( v21 )
    {
      v22 = sub_25394D0(*(const __m128i **)(v16 + 40), (__int64)&v49);
      v21 = (unsigned __int64)v22;
      do
      {
        v23 = v22;
        v22 = (__m128i *)v22[1].m128i_i64[0];
      }
      while ( v22 );
      v50 = v23;
      v24 = v21;
      do
      {
        v25 = (__int128 *)v24;
        v24 = *(_QWORD *)(v24 + 24);
      }
      while ( v24 );
      v51 = v25;
      v26 = *(_QWORD *)(v16 + 64);
      *((_QWORD *)&v49 + 1) = v21;
      v52 = v26;
    }
    v46 = &unk_4A16DD8;
    sub_255C230((__int64)&v48 + 8, v21);
    v27 = *(_DWORD **)(a1 + 24);
    v28 = *(__int64 (**)(void))(*(_QWORD *)v27 + 16LL);
    if ( (char *)v28 == (char *)sub_2505DB0 )
      LOBYTE(v9) = v27[5] != 0;
    else
      LODWORD(v9) = v28();
  }
  else
  {
    LODWORD(v9) = 0;
  }
  if ( v43 != &v45 )
    _libc_free((unsigned __int64)v43);
  return (unsigned int)v9;
}
