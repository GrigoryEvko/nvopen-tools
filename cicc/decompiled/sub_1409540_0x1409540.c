// Function: sub_1409540
// Address: 0x1409540
//
__int64 __fastcall sub_1409540(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 i; // r15
  __int64 v11; // r13
  __int64 v12; // rdi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // r12
  int v15; // r8d
  unsigned int v16; // esi
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r10
  unsigned int v20; // eax
  __int64 v21; // rdx
  __int64 v22; // rdi
  const __m128i *v23; // r9
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v27; // rax
  unsigned __int8 v28; // al
  unsigned __int64 v29; // rsi
  __int64 v30; // r14
  __int64 *v31; // rax
  _BYTE *v32; // rdi
  __int64 v33; // r12
  const __m128i *v34; // r15
  __int64 *v35; // rbx
  unsigned __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 v39; // r9
  int v40; // eax
  _QWORD *v41; // rax
  __int64 v42; // r12
  __int64 *v43; // rax
  __int64 *v44; // r15
  __int64 *v45; // rbx
  const __m128i *v46; // r12
  unsigned __int64 v47; // rdi
  __int64 v48; // rsi
  __int64 v49; // rax
  int v50; // r14d
  int v51; // r14d
  __int64 v52; // r10
  unsigned int v53; // esi
  __int64 v54; // rdi
  int v55; // r11d
  __int64 v56; // r9
  int v57; // r14d
  int v58; // r14d
  __int64 v59; // r9
  __int64 v60; // r10
  unsigned int v61; // esi
  int v62; // r11d
  __int64 v63; // [rsp+8h] [rbp-E8h]
  __int64 v64; // [rsp+10h] [rbp-E0h]
  __int64 v65; // [rsp+10h] [rbp-E0h]
  int v66; // [rsp+18h] [rbp-D8h]
  __int64 v67; // [rsp+18h] [rbp-D8h]
  int v68; // [rsp+18h] [rbp-D8h]
  int v69; // [rsp+18h] [rbp-D8h]
  __int64 v70; // [rsp+20h] [rbp-D0h]
  __int64 v72; // [rsp+38h] [rbp-B8h] BYREF
  __m128i v73; // [rsp+40h] [rbp-B0h] BYREF
  __m128i v74; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE v75[144]; // [rsp+60h] [rbp-90h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 160) = a2;
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_101:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F99308 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_101;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F99308);
  v6 = *(_QWORD *)(a2 + 80);
  v70 = v5 + 160;
  if ( a2 + 72 == v6 )
  {
    v7 = 0;
  }
  else
  {
    if ( !v6 )
      BUG();
    while ( 1 )
    {
      v7 = *(_QWORD *)(v6 + 24);
      if ( v7 != v6 + 16 )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( a2 + 72 == v6 )
        break;
      if ( !v6 )
        BUG();
    }
  }
  v8 = a2 + 72;
LABEL_12:
  if ( v6 != v8 )
  {
    v9 = v8;
    i = v7;
    v11 = v9;
    do
    {
      v12 = i - 24;
      if ( !i )
        v12 = 0;
      v72 = v12;
      if ( (unsigned __int8)sub_15F2ED0(v12) || (unsigned __int8)sub_15F3040(v72) )
      {
        v13 = sub_141C430(v70, v72, 0);
        v14 = v13;
        v15 = v13 & 7;
        if ( v15 != 3 || v13 >> 61 != 1 )
        {
          v16 = *(_DWORD *)(a1 + 192);
          v17 = a1 + 168;
          if ( v16 )
          {
            v18 = v72;
            v19 = *(_QWORD *)(a1 + 176);
            v20 = (v16 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
            v21 = v19 + 168LL * v20;
            v22 = *(_QWORD *)v21;
            v23 = (const __m128i *)(v21 + 8);
            if ( v72 == *(_QWORD *)v21 )
              goto LABEL_21;
            v66 = 1;
            v39 = 0;
            while ( v22 != -8 )
            {
              if ( v22 == -16 && !v39 )
                v39 = v21;
              v20 = (v16 - 1) & (v66 + v20);
              v21 = v19 + 168LL * v20;
              v22 = *(_QWORD *)v21;
              if ( v72 == *(_QWORD *)v21 )
              {
                v23 = (const __m128i *)(v21 + 8);
                goto LABEL_21;
              }
              ++v66;
            }
            if ( v39 )
              v21 = v39;
            ++*(_QWORD *)(a1 + 168);
            v40 = *(_DWORD *)(a1 + 184) + 1;
            if ( 4 * v40 < 3 * v16 )
            {
              if ( v16 - *(_DWORD *)(a1 + 188) - v40 <= v16 >> 3 )
              {
                v69 = v15;
                sub_1408330(v17, v16);
                v57 = *(_DWORD *)(a1 + 192);
                if ( !v57 )
                {
LABEL_100:
                  ++*(_DWORD *)(a1 + 184);
                  BUG();
                }
                v58 = v57 - 1;
                v59 = 0;
                v60 = *(_QWORD *)(a1 + 176);
                v15 = v69;
                v61 = v58 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
                v21 = v60 + 168LL * v61;
                v62 = 1;
                v18 = *(_QWORD *)v21;
                v40 = *(_DWORD *)(a1 + 184) + 1;
                if ( v72 != *(_QWORD *)v21 )
                {
                  while ( v18 != -8 )
                  {
                    if ( !v59 && v18 == -16 )
                      v59 = v21;
                    v61 = v58 & (v62 + v61);
                    v21 = v60 + 168LL * v61;
                    v18 = *(_QWORD *)v21;
                    if ( v72 == *(_QWORD *)v21 )
                      goto LABEL_53;
                    ++v62;
                  }
                  v18 = v72;
                  if ( v59 )
                    v21 = v59;
                }
              }
              goto LABEL_53;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 168);
          }
          v68 = v15;
          sub_1408330(v17, 2 * v16);
          v50 = *(_DWORD *)(a1 + 192);
          if ( !v50 )
            goto LABEL_100;
          v18 = v72;
          v51 = v50 - 1;
          v52 = *(_QWORD *)(a1 + 176);
          v15 = v68;
          v53 = v51 & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
          v21 = v52 + 168LL * v53;
          v54 = *(_QWORD *)v21;
          v40 = *(_DWORD *)(a1 + 184) + 1;
          if ( *(_QWORD *)v21 != v72 )
          {
            v55 = 1;
            v56 = 0;
            while ( v54 != -8 )
            {
              if ( v54 == -16 && !v56 )
                v56 = v21;
              v53 = v51 & (v55 + v53);
              v21 = v52 + 168LL * v53;
              v54 = *(_QWORD *)v21;
              if ( v72 == *(_QWORD *)v21 )
                goto LABEL_53;
              ++v55;
            }
            if ( v56 )
              v21 = v56;
          }
LABEL_53:
          *(_DWORD *)(a1 + 184) = v40;
          if ( *(_QWORD *)v21 != -8 )
            --*(_DWORD *)(a1 + 188);
          v23 = (const __m128i *)(v21 + 8);
          *(_QWORD *)v21 = v18;
          memset((void *)(v21 + 8), 0, 0xA0u);
          *(_BYTE *)(v21 + 16) = 1;
          v41 = (_QWORD *)(v21 + 24);
          do
          {
            if ( v41 )
            {
              *v41 = -2;
              v41[1] = -8;
            }
            v41 += 2;
          }
          while ( (_QWORD *)(v21 + 88) != v41 );
          *(_QWORD *)(v21 + 88) = v21 + 104;
          *(_QWORD *)(v21 + 96) = 0x400000000LL;
LABEL_21:
          v24 = v14 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v15 != 1 )
          {
            if ( v15 == 2 )
            {
              v24 |= 2u;
            }
            else
            {
              v24 |= 6u;
              if ( v15 == 3 )
                v24 = 2LL * (v14 >> 61 != 2) + 4;
            }
          }
          v74 = (__m128i)v24;
          sub_14090C0(v23, &v74);
          goto LABEL_26;
        }
        v28 = *(_BYTE *)(v72 + 16);
        if ( v28 <= 0x17u )
        {
          v30 = a1 + 168;
LABEL_40:
          v74.m128i_i64[0] = (__int64)v75;
          v74.m128i_i64[1] = 0x400000000LL;
          sub_141E480(v70, v72, &v74, 0);
          v31 = sub_1408E20(v30, &v72);
          v32 = (_BYTE *)(v74.m128i_i64[0] + 24LL * v74.m128i_u32[2]);
          if ( (_BYTE *)v74.m128i_i64[0] != v32 )
          {
            v64 = i;
            v33 = v74.m128i_i64[0] + 24LL * v74.m128i_u32[2];
            v63 = v6;
            v34 = (const __m128i *)(v31 + 1);
            v35 = (__int64 *)v74.m128i_i64[0];
            do
            {
              v36 = v35[1];
              v37 = *v35;
              v35 += 3;
              v38 = sub_1408850(v36);
              v73.m128i_i64[1] = v37;
              v73.m128i_i64[0] = v38;
              sub_14090C0(v34, &v73);
            }
            while ( (__int64 *)v33 != v35 );
            i = v64;
            v6 = v63;
            v32 = (_BYTE *)v74.m128i_i64[0];
          }
          if ( v32 != v75 )
            _libc_free((unsigned __int64)v32);
          goto LABEL_26;
        }
        v29 = v72 | 4;
        if ( v28 != 78 )
        {
          v30 = a1 + 168;
          if ( v28 != 29 )
            goto LABEL_40;
          v29 = v72 & 0xFFFFFFFFFFFFFFFBLL;
        }
        v30 = a1 + 168;
        if ( (v29 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_40;
        v42 = sub_1418110(v70);
        v43 = sub_1408E20(v30, &v72) + 1;
        if ( *(_QWORD *)(v42 + 8) != *(_QWORD *)v42 )
        {
          v67 = i;
          v44 = *(__int64 **)(v42 + 8);
          v65 = v6;
          v45 = *(__int64 **)v42;
          v46 = (const __m128i *)v43;
          do
          {
            v47 = v45[1];
            v48 = *v45;
            v45 += 2;
            v49 = sub_1408850(v47);
            v74.m128i_i64[1] = v48;
            v74.m128i_i64[0] = v49;
            sub_14090C0(v46, &v74);
          }
          while ( v44 != v45 );
          i = v67;
          v6 = v65;
        }
      }
LABEL_26:
      for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v6 + 24) )
      {
        v25 = v6 - 24;
        if ( !v6 )
          v25 = 0;
        if ( i != v25 + 40 )
          break;
        v6 = *(_QWORD *)(v6 + 8);
        if ( v11 == v6 )
        {
          v27 = v11;
          v7 = i;
          v8 = v27;
          goto LABEL_12;
        }
        if ( !v6 )
          BUG();
      }
    }
    while ( v6 != v11 );
  }
  return 0;
}
