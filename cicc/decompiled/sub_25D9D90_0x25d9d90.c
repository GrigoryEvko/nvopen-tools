// Function: sub_25D9D90
// Address: 0x25d9d90
//
void __fastcall sub_25D9D90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v5; // r12
  _QWORD *v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  _QWORD *v10; // rbx
  unsigned int v11; // esi
  __int32 v12; // r11d
  __int64 v13; // r9
  unsigned int v14; // r8d
  __int64 *v15; // rdi
  __int64 *v16; // rdx
  __int64 v17; // rcx
  bool v18; // zf
  __int64 v19; // r15
  __int64 v20; // rdx
  unsigned __int8 v21; // al
  __int64 v22; // rdx
  int v23; // edi
  int v24; // ecx
  unsigned __int64 v25; // rdx
  __m128i *v26; // rcx
  __m128i *v27; // r8
  __m128i *v28; // rax
  _QWORD *v29; // rbx
  unsigned __int64 v30; // rdx
  __m128i v31; // xmm0
  int v32; // eax
  __int64 *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 *v37; // rax
  _QWORD *v38; // r12
  __m128i *v39; // r15
  __m128i *v40; // r13
  _QWORD *v41; // rdx
  _QWORD *v42; // rsi
  int v43; // esi
  int v44; // esi
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 v47; // rdi
  int v48; // r11d
  int v49; // esi
  unsigned int v50; // esi
  __int64 v51; // r8
  int v52; // r11d
  __int64 v53; // rax
  __int64 v54; // rdi
  _QWORD *v55; // [rsp+8h] [rbp-A8h]
  __m128i v56; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v57; // [rsp+20h] [rbp-90h]
  __int64 v58; // [rsp+28h] [rbp-88h]
  _BYTE **v59; // [rsp+30h] [rbp-80h]
  __int64 v60; // [rsp+38h] [rbp-78h]
  _QWORD *v61; // [rsp+40h] [rbp-70h]
  __int64 v62; // [rsp+48h] [rbp-68h]
  __m128i v63; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v64; // [rsp+60h] [rbp-50h] BYREF
  int v65; // [rsp+68h] [rbp-48h]
  int v66; // [rsp+6Ch] [rbp-44h]
  _BYTE v67[64]; // [rsp+70h] [rbp-40h] BYREF

  v64 = v67;
  v2 = a2 + 8;
  v3 = *(_QWORD *)(a2 + 16);
  v66 = 2;
  v60 = v2;
  v62 = v3;
  v59 = &v64;
  if ( v3 == v2 )
    return;
  while ( 2 )
  {
    v65 = 0;
    v5 = v62 - 56;
    if ( !v62 )
      v5 = 0;
    sub_B91D10(v5, 19, (__int64)v59);
    if ( sub_B2FC80(v5) || !v65 )
      goto LABEL_3;
    v6 = v64;
    v61 = &v64[8 * v65];
    v58 = a1 + 440;
    do
    {
      while ( 1 )
      {
        v20 = *v6;
        v21 = *(_BYTE *)(*v6 - 16LL);
        if ( (v21 & 2) != 0 )
        {
          v7 = *(_QWORD **)(v20 - 32);
          v8 = v7[1];
        }
        else
        {
          v22 = v20 - 8LL * ((v21 >> 2) & 0xF);
          v8 = *(_QWORD *)(v22 - 8);
          v7 = (_QWORD *)(v22 - 16);
        }
        v9 = *(_QWORD *)(*v7 + 136LL);
        v10 = *(_QWORD **)(v9 + 24);
        if ( *(_DWORD *)(v9 + 32) > 0x40u )
          v10 = (_QWORD *)*v10;
        v11 = *(_DWORD *)(a1 + 464);
        if ( !v11 )
        {
          ++*(_QWORD *)(a1 + 440);
          goto LABEL_63;
        }
        v12 = 1;
        v13 = *(_QWORD *)(a1 + 448);
        v14 = (v11 - 1) & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
        v15 = (__int64 *)(v13 + 136LL * v14);
        v16 = 0;
        v17 = *v15;
        if ( v8 != *v15 )
        {
          while ( v17 != -4096 )
          {
            if ( !v16 && v17 == -8192 )
              v16 = v15;
            v14 = (v11 - 1) & (v12 + v14);
            v56.m128i_i32[0] = v12 + 1;
            v15 = (__int64 *)(v13 + 136LL * v14);
            v17 = *v15;
            if ( *v15 == v8 )
              goto LABEL_14;
            v12 = v56.m128i_i32[0];
          }
          if ( !v16 )
            v16 = v15;
          v23 = *(_DWORD *)(a1 + 456);
          ++*(_QWORD *)(a1 + 440);
          v24 = v23 + 1;
          if ( 4 * (v23 + 1) < 3 * v11 )
          {
            if ( v11 - *(_DWORD *)(a1 + 460) - v24 > v11 >> 3 )
              goto LABEL_29;
            v56.m128i_i32[0] = ((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9);
            sub_25D8AF0(v58, v11);
            v49 = *(_DWORD *)(a1 + 464);
            if ( !v49 )
            {
LABEL_86:
              ++*(_DWORD *)(a1 + 456);
              BUG();
            }
            v50 = v49 - 1;
            v51 = *(_QWORD *)(a1 + 448);
            v13 = 0;
            v52 = 1;
            LODWORD(v53) = v50 & v56.m128i_i32[0];
            v16 = (__int64 *)(v51 + 136LL * (v50 & v56.m128i_i32[0]));
            v54 = *v16;
            v24 = *(_DWORD *)(a1 + 456) + 1;
            if ( v8 == *v16 )
              goto LABEL_29;
            while ( v54 != -4096 )
            {
              if ( !v13 && v54 == -8192 )
                v13 = (__int64)v16;
              v53 = v50 & ((_DWORD)v53 + v52);
              v16 = (__int64 *)(v51 + 136 * v53);
              v54 = *v16;
              if ( *v16 == v8 )
                goto LABEL_29;
              ++v52;
            }
            goto LABEL_67;
          }
LABEL_63:
          sub_25D8AF0(v58, 2 * v11);
          v43 = *(_DWORD *)(a1 + 464);
          if ( !v43 )
            goto LABEL_86;
          v44 = v43 - 1;
          v45 = *(_QWORD *)(a1 + 448);
          LODWORD(v46) = v44 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v16 = (__int64 *)(v45 + 136LL * (unsigned int)v46);
          v47 = *v16;
          v24 = *(_DWORD *)(a1 + 456) + 1;
          if ( v8 == *v16 )
            goto LABEL_29;
          v48 = 1;
          v13 = 0;
          while ( v47 != -4096 )
          {
            if ( !v13 && v47 == -8192 )
              v13 = (__int64)v16;
            v46 = v44 & (unsigned int)(v46 + v48);
            v16 = (__int64 *)(v45 + 136 * v46);
            v47 = *v16;
            if ( *v16 == v8 )
              goto LABEL_29;
            ++v48;
          }
LABEL_67:
          if ( v13 )
            v16 = (__int64 *)v13;
LABEL_29:
          *(_DWORD *)(a1 + 456) = v24;
          if ( *v16 != -4096 )
            --*(_DWORD *)(a1 + 460);
          *v16 = v8;
          memset(v16 + 1, 0, 0x80u);
          v63.m128i_i64[0] = v5;
          v19 = (__int64)(v16 + 1);
          v16[1] = (__int64)(v16 + 3);
          v16[2] = 0x400000000LL;
          v16[14] = (__int64)(v16 + 12);
          v16[15] = (__int64)(v16 + 12);
          v63.m128i_i64[1] = (__int64)v10;
          goto LABEL_32;
        }
LABEL_14:
        v18 = v15[16] == 0;
        v63.m128i_i64[0] = v5;
        v19 = (__int64)(v15 + 1);
        v63.m128i_i64[1] = (__int64)v10;
        if ( !v18 )
        {
          sub_25D9C00((__int64)(v15 + 11), &v63);
          goto LABEL_16;
        }
LABEL_32:
        v25 = *(unsigned int *)(v19 + 8);
        v26 = *(__m128i **)v19;
        v27 = (__m128i *)(*(_QWORD *)v19 + 16 * v25);
        if ( *(__m128i **)v19 != v27 )
          break;
        if ( v25 <= 3 )
          goto LABEL_39;
        v29 = (_QWORD *)(v19 + 80);
LABEL_58:
        *(_DWORD *)(v19 + 8) = 0;
        sub_25D9C00((__int64)v29, &v63);
LABEL_16:
        if ( v61 == ++v6 )
          goto LABEL_42;
      }
      v28 = *(__m128i **)v19;
      do
      {
        if ( v5 == v28->m128i_i64[0] && v10 == (_QWORD *)v28->m128i_i64[1] )
        {
          if ( v27 != v28 )
            goto LABEL_16;
          v29 = (_QWORD *)(v19 + 80);
          if ( v25 <= 3 )
            goto LABEL_39;
LABEL_53:
          v56.m128i_i64[0] = v5;
          v57 = v19;
          v38 = (_QWORD *)(v19 + 88);
          v39 = v27;
          v55 = v6;
          v40 = v26;
          do
          {
            v42 = sub_25D9C40(v29, v38, (unsigned __int64 *)v40);
            if ( v41 )
              sub_25D6B40((__int64)v29, (__int64)v42, v41, v40);
            ++v40;
          }
          while ( v39 != v40 );
          v5 = v56.m128i_i64[0];
          v19 = v57;
          v6 = v55;
          goto LABEL_58;
        }
        ++v28;
      }
      while ( v27 != v28 );
      v29 = (_QWORD *)(v19 + 80);
      if ( v25 > 3 )
        goto LABEL_53;
LABEL_39:
      v30 = v25 + 1;
      v31 = _mm_load_si128(&v63);
      if ( v30 > *(unsigned int *)(v19 + 12) )
      {
        v56 = v31;
        sub_C8D5F0(v19, (const void *)(v19 + 16), v30, 0x10u, (__int64)v27, v13);
        v31 = _mm_load_si128(&v56);
        v27 = (__m128i *)(*(_QWORD *)v19 + 16LL * *(unsigned int *)(v19 + 8));
      }
      *v27 = v31;
      ++v6;
      ++*(_DWORD *)(v19 + 8);
    }
    while ( v61 != v6 );
LABEL_42:
    if ( !v5 )
      goto LABEL_3;
    v32 = sub_B92110(v5);
    if ( v32 != 2 && (!*(_BYTE *)a1 || v32 != 1) )
      goto LABEL_3;
    if ( !*(_BYTE *)(a1 + 500) )
    {
LABEL_80:
      sub_C8CC70(a1 + 472, v5, (__int64)v33, v34, v35, v36);
      goto LABEL_3;
    }
    v37 = *(__int64 **)(a1 + 480);
    v34 = *(unsigned int *)(a1 + 492);
    v33 = &v37[v34];
    if ( v37 == v33 )
    {
LABEL_50:
      if ( (unsigned int)v34 < *(_DWORD *)(a1 + 488) )
      {
        *(_DWORD *)(a1 + 492) = v34 + 1;
        *v33 = v5;
        ++*(_QWORD *)(a1 + 472);
        goto LABEL_3;
      }
      goto LABEL_80;
    }
    while ( v5 != *v37 )
    {
      if ( v33 == ++v37 )
        goto LABEL_50;
    }
LABEL_3:
    v62 = *(_QWORD *)(v62 + 8);
    if ( v60 != v62 )
      continue;
    break;
  }
  if ( v64 != v67 )
    _libc_free((unsigned __int64)v64);
}
