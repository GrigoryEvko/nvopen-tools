// Function: sub_2DC74F0
// Address: 0x2dc74f0
//
__int64 __fastcall sub_2DC74F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  __int64 v7; // rdi
  __int64 (*v8)(); // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r13
  __m128i v13; // xmm5
  __m128i v14; // xmm6
  __m128i v15; // xmm7
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  unsigned int v18; // eax
  _QWORD **v19; // r14
  _QWORD **i; // r13
  __int64 v21; // rax
  _QWORD *v22; // r12
  unsigned __int64 v23; // r15
  __int64 v24; // rdi
  __m128i v25; // xmm1
  __m128i v26; // xmm2
  __m128i v27; // xmm3
  __m128i v28; // xmm4
  unsigned int v29; // eax
  _QWORD *v30; // r13
  _QWORD *v31; // r14
  __int64 v32; // rdi
  __int64 *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 *v38; // rdx
  __int64 *v39; // r15
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 *v42; // r13
  __int64 v43; // r14
  __int64 *v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  unsigned int v51; // r12d
  __int64 **v53; // rax
  __int64 **v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // r11
  __int64 v57; // r10
  __int64 v58; // [rsp+0h] [rbp-110h]
  __int64 v59; // [rsp+8h] [rbp-108h]
  __int64 v60; // [rsp+10h] [rbp-100h]
  __int64 v61; // [rsp+10h] [rbp-100h]
  __int64 *v62; // [rsp+18h] [rbp-F8h]
  __m128i v63; // [rsp+30h] [rbp-E0h] BYREF
  __m128i v64; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v65; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v66; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v67; // [rsp+70h] [rbp-A0h] BYREF
  _BYTE v68[8]; // [rsp+80h] [rbp-90h] BYREF
  unsigned __int64 v69; // [rsp+88h] [rbp-88h]
  unsigned int v70; // [rsp+94h] [rbp-7Ch]
  unsigned int v71; // [rsp+98h] [rbp-78h]
  unsigned __int8 v72; // [rsp+9Ch] [rbp-74h]
  __int64 v73; // [rsp+A8h] [rbp-68h]
  unsigned __int64 v74; // [rsp+B8h] [rbp-58h]
  int v75; // [rsp+C4h] [rbp-4Ch]
  __int64 v76; // [rsp+C8h] [rbp-48h]
  unsigned int v77; // [rsp+D8h] [rbp-38h]

  v3 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_5027190);
  if ( !v3 )
    return 0;
  v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_5027190);
  if ( !v4 )
    return 0;
  v5 = *(_QWORD *)(v4 + 256);
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 16LL);
  if ( v6 == sub_23CE270 )
    BUG();
  v7 = ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a2);
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 144LL);
  if ( v8 != sub_2C8F680 )
    ((void (__fastcall *)(__int64))v8)(v7);
  v9 = *(__int64 **)(a1 + 8);
  v10 = *v9;
  v11 = v9[1];
  if ( v10 == v11 )
LABEL_77:
    BUG();
  while ( *(_UNKNOWN **)v10 != &unk_4F6D3F0 )
  {
    v10 += 16;
    if ( v11 == v10 )
      goto LABEL_77;
  }
  v12 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v10 + 8) + 104LL))(*(_QWORD *)(v10 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v68);
  sub_983BD0((__int64)&v63, v12 + 176, a2);
  v62 = (__int64 *)(v12 + 408);
  if ( *(_BYTE *)(v12 + 488) )
  {
    v25 = _mm_loadu_si128(&v64);
    v26 = _mm_loadu_si128(&v65);
    v27 = _mm_loadu_si128(&v66);
    v28 = _mm_loadu_si128(&v67);
    *(__m128i *)(v12 + 408) = _mm_loadu_si128(&v63);
    *(__m128i *)(v12 + 424) = v25;
    *(__m128i *)(v12 + 440) = v26;
    *(__m128i *)(v12 + 456) = v27;
    *(__m128i *)(v12 + 472) = v28;
  }
  else
  {
    v13 = _mm_loadu_si128(&v63);
    v14 = _mm_loadu_si128(&v64);
    *(_BYTE *)(v12 + 488) = 1;
    v15 = _mm_loadu_si128(&v65);
    v16 = _mm_loadu_si128(&v66);
    v17 = _mm_loadu_si128(&v67);
    *(__m128i *)(v12 + 408) = v13;
    *(__m128i *)(v12 + 424) = v14;
    *(__m128i *)(v12 + 440) = v15;
    *(__m128i *)(v12 + 456) = v16;
    *(__m128i *)(v12 + 472) = v17;
  }
  sub_C7D6A0(v76, 24LL * v77, 8);
  v18 = v74;
  if ( (_DWORD)v74 )
  {
    v19 = (_QWORD **)(v73 + 32LL * (unsigned int)v74);
    for ( i = (_QWORD **)(v73 + 8); ; i += 4 )
    {
      v21 = (__int64)*(i - 1);
      if ( v21 != -4096 && v21 != -8192 )
      {
        v22 = *i;
        while ( v22 != i )
        {
          v23 = (unsigned __int64)v22;
          v22 = (_QWORD *)*v22;
          v24 = *(_QWORD *)(v23 + 24);
          if ( v24 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v24 + 8LL))(v24);
          j_j___libc_free_0(v23);
        }
      }
      if ( v19 == i + 3 )
        break;
    }
    v18 = v74;
  }
  sub_C7D6A0(v73, 32LL * v18, 8);
  v29 = v71;
  if ( v71 )
  {
    v30 = (_QWORD *)v69;
    v31 = (_QWORD *)(v69 + 16LL * v71);
    do
    {
      if ( *v30 != -8192 && *v30 != -4096 )
      {
        v32 = v30[1];
        if ( v32 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v32 + 8LL))(v32);
      }
      v30 += 2;
    }
    while ( v31 != v30 );
    v29 = v71;
  }
  sub_C7D6A0(v69, 16LL * v29, 8);
  v33 = *(__int64 **)(a1 + 8);
  v34 = *v33;
  v35 = v33[1];
  if ( v34 == v35 )
LABEL_76:
    BUG();
  while ( *(_UNKNOWN **)v34 != &unk_4F89C28 )
  {
    v34 += 16;
    if ( v35 == v34 )
      goto LABEL_76;
  }
  v36 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v34 + 8) + 104LL))(*(_QWORD *)(v34 + 8), &unk_4F89C28);
  v37 = sub_DFED00(v36, a2);
  v38 = *(__int64 **)(a1 + 8);
  v39 = (__int64 *)v37;
  v40 = *v38;
  v41 = v38[1];
  if ( v40 == v41 )
LABEL_74:
    BUG();
  while ( *(_UNKNOWN **)v40 != &unk_4F87C64 )
  {
    v40 += 16;
    if ( v41 == v40 )
      goto LABEL_74;
  }
  v42 = 0;
  v43 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v40 + 8) + 104LL))(
                      *(_QWORD *)(v40 + 8),
                      &unk_4F87C64)
                  + 176);
  if ( v43 )
  {
    v42 = *(__int64 **)(v43 + 8);
    if ( v42 )
    {
      v44 = *(__int64 **)(a1 + 8);
      v45 = *v44;
      v46 = v44[1];
      if ( v45 == v46 )
LABEL_75:
        BUG();
      while ( *(_UNKNOWN **)v45 != &unk_4F8EE48 )
      {
        v45 += 16;
        if ( v46 == v45 )
          goto LABEL_75;
      }
      v47 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v45 + 8) + 104LL))(
              *(_QWORD *)(v45 + 8),
              &unk_4F8EE48);
      v42 = (__int64 *)(v47 + 176);
      if ( !*(_BYTE *)(v47 + 184) )
      {
        v55 = *(_QWORD *)(v47 + 200);
        v56 = *(_QWORD *)(v47 + 208);
        v57 = *(_QWORD *)(v55 + 176);
        if ( !*(_BYTE *)(v57 + 280) )
        {
          v58 = v47;
          v59 = *(_QWORD *)(v47 + 208);
          v61 = *(_QWORD *)(v55 + 176);
          sub_FF9360((_QWORD *)v61, *(_QWORD *)(v57 + 288), *(_QWORD *)(v57 + 296), *(__int64 **)(v57 + 304), 0, 0);
          v57 = v61;
          v47 = v58;
          v56 = v59;
          *(_BYTE *)(v61 + 280) = 1;
        }
        v60 = v47;
        sub_FE7D70(v42, *(const char **)(v47 + 192), v57, v56);
        *(_BYTE *)(v60 + 184) = 1;
      }
    }
  }
  v48 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8144C);
  if ( v48 && (v49 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v48 + 104LL))(v48, &unk_4F8144C)) != 0 )
    v50 = v49 + 176;
  else
    v50 = 0;
  v51 = 1;
  sub_2DC4260((__int64)v68, a2, v62, v39, v43, v42, v50);
  if ( v75 != (_DWORD)v76 )
  {
LABEL_51:
    if ( BYTE4(v76) )
      goto LABEL_52;
    goto LABEL_62;
  }
  v51 = v72;
  if ( !v72 )
  {
    LOBYTE(v51) = sub_C8CA60((__int64)v68, (__int64)&qword_4F82400) == 0;
    goto LABEL_51;
  }
  v53 = (__int64 **)v69;
  v54 = (__int64 **)(v69 + 8LL * v70);
  if ( (__int64 **)v69 != v54 )
  {
    while ( *v53 != &qword_4F82400 )
    {
      if ( v54 == ++v53 )
        goto LABEL_61;
    }
    v51 = 0;
  }
LABEL_61:
  if ( !BYTE4(v76) )
  {
LABEL_62:
    _libc_free(v74);
LABEL_52:
    if ( !v72 )
      _libc_free(v69);
  }
  return v51;
}
