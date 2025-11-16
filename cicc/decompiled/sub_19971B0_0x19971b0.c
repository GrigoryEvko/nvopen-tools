// Function: sub_19971B0
// Address: 0x19971b0
//
void __fastcall sub_19971B0(__int64 a1)
{
  __int64 *v1; // r14
  __int64 *v2; // r8
  __int64 *v3; // rbx
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int64 v7; // rcx
  unsigned __int64 v8; // rdx
  __int64 *v9; // r10
  __int64 *v10; // r12
  __int64 *v11; // r14
  __int64 *v12; // r15
  __int64 v13; // r13
  __int64 *v14; // rbx
  __int64 *v15; // rax
  __int64 v16; // r15
  __int64 v17; // r13
  _QWORD *v18; // rdx
  _QWORD *v19; // rax
  _QWORD *v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r12
  char v24; // r10
  __int64 v25; // r8
  _QWORD *v26; // rdi
  _QWORD *v27; // rsi
  __int64 *v28; // r8
  __int64 v29; // rdx
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 *v32; // rax
  __int64 v33; // r11
  unsigned __int64 v34; // rax
  unsigned int v35; // ebx
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rsi
  unsigned int v39; // ecx
  _QWORD *v40; // rax
  _QWORD *v41; // r12
  __int64 v42; // r13
  int v43; // eax
  __int64 *v44; // rdi
  __int64 *v45; // rax
  __int64 *v46; // rsi
  unsigned int v47; // edx
  __int64 v48; // rcx
  unsigned int v49; // edi
  __int64 *v50; // rax
  __int64 v51; // r10
  unsigned __int64 v52; // rdi
  _QWORD *v53; // rax
  __int64 v54; // rsi
  __int64 v55; // r12
  unsigned int v56; // r13d
  int v57; // eax
  _QWORD *v58; // rdx
  __int64 *v59; // rdi
  __int64 *v60; // rcx
  int v61; // esi
  int v62; // esi
  __int64 *v63; // [rsp+8h] [rbp-C8h]
  __int64 v64; // [rsp+10h] [rbp-C0h]
  __int64 *v65; // [rsp+10h] [rbp-C0h]
  __int64 *v66; // [rsp+18h] [rbp-B8h]
  unsigned int v67; // [rsp+20h] [rbp-B0h]
  __int64 v68; // [rsp+20h] [rbp-B0h]
  __int64 v70; // [rsp+30h] [rbp-A0h]
  __int64 v71; // [rsp+30h] [rbp-A0h]
  __int64 *v72; // [rsp+38h] [rbp-98h]
  __int64 v73; // [rsp+38h] [rbp-98h]
  __int64 v74; // [rsp+48h] [rbp-88h] BYREF
  __int64 v75; // [rsp+50h] [rbp-80h] BYREF
  __int64 *v76; // [rsp+58h] [rbp-78h]
  __int64 *v77; // [rsp+60h] [rbp-70h]
  __int64 v78; // [rsp+68h] [rbp-68h]
  int v79; // [rsp+70h] [rbp-60h]
  _BYTE v80[88]; // [rsp+78h] [rbp-58h] BYREF

  v1 = &v74;
  v2 = (__int64 *)v80;
  v3 = (__int64 *)v80;
  v4 = *(unsigned int *)(a1 + 376);
  v75 = 0;
  v76 = (__int64 *)v80;
  v77 = (__int64 *)v80;
  v78 = 4;
  v79 = 0;
  while ( 2 )
  {
    v5 = *(_QWORD *)(a1 + 368);
    v6 = v5 + 1984 * v4;
    if ( v5 == v6 )
      break;
    v7 = 1;
    while ( 1 )
    {
      v8 = *(unsigned int *)(v5 + 752);
      if ( v8 > 0xFFFE )
        break;
      v7 *= v8;
      if ( v7 > 0x3FFFB )
        break;
      v5 += 1984;
      if ( v6 == v5 )
      {
        if ( v7 <= 0xFFFE )
          goto LABEL_94;
        break;
      }
    }
    v9 = *(__int64 **)(a1 + 32160);
    v72 = &v9[*(unsigned int *)(a1 + 32168)];
    if ( v72 == v9 )
      break;
    v66 = v1;
    v10 = v3;
    v11 = *(__int64 **)(a1 + 32160);
    v70 = 0;
    v12 = v2;
    v67 = 0;
    do
    {
      while ( 1 )
      {
        v13 = *v11;
        if ( *(_WORD *)(*v11 + 24) != 7 )
          goto LABEL_9;
        if ( v10 == v12 )
        {
          v14 = &v10[HIDWORD(v78)];
          if ( v14 == v10 )
          {
            v36 = (__int64)v10;
            v15 = v10;
          }
          else
          {
            v15 = v10;
            do
            {
              if ( v13 == *v15 )
                break;
              ++v15;
            }
            while ( v14 != v15 );
            v36 = (__int64)&v10[HIDWORD(v78)];
          }
        }
        else
        {
          v14 = &v12[(unsigned int)v78];
          v15 = sub_16CC9F0((__int64)&v75, *v11);
          if ( v13 == *v15 )
          {
            v12 = v77;
            v10 = v76;
            v36 = (__int64)(v77 == v76 ? &v77[HIDWORD(v78)] : &v77[(unsigned int)v78]);
          }
          else
          {
            v12 = v77;
            v10 = v76;
            if ( v77 != v76 )
            {
              v15 = &v77[(unsigned int)v78];
              goto LABEL_15;
            }
            v15 = &v77[HIDWORD(v78)];
            v36 = (__int64)v15;
          }
        }
        while ( (__int64 *)v36 != v15 && (unsigned __int64)*v15 >= 0xFFFFFFFFFFFFFFFELL )
          ++v15;
LABEL_15:
        if ( v14 != v15 )
          break;
        if ( !*(_WORD *)(**(_QWORD **)(v13 + 32) + 24LL) )
        {
          if ( v12 != v10 )
            goto LABEL_89;
          v59 = &v10[HIDWORD(v78)];
          if ( v59 == v10 )
          {
LABEL_116:
            if ( HIDWORD(v78) >= (unsigned int)v78 )
            {
LABEL_89:
              sub_16CCBA0((__int64)&v75, v13);
              goto LABEL_90;
            }
            ++HIDWORD(v78);
            *v59 = v13;
            v10 = v76;
            ++v75;
          }
          else
          {
            v60 = 0;
            while ( v13 != *v10 )
            {
              if ( *v10 == -2 )
                v60 = v10;
              if ( v59 == ++v10 )
              {
                if ( !v60 )
                  goto LABEL_116;
                *v60 = v13;
                v10 = v76;
                --v79;
                ++v75;
                goto LABEL_91;
              }
            }
LABEL_90:
            v10 = v76;
          }
LABEL_91:
          v12 = v77;
          goto LABEL_9;
        }
        if ( !v70 )
        {
          v47 = *(_DWORD *)(a1 + 32152);
          v48 = *(_QWORD *)(a1 + 32136);
          if ( v47 )
          {
            v49 = (v47 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v50 = (__int64 *)(v48 + 16LL * v49);
            v51 = *v50;
            if ( v13 == *v50 )
              goto LABEL_86;
            v57 = 1;
            while ( v51 != -8 )
            {
              v62 = v57 + 1;
              v49 = (v47 - 1) & (v57 + v49);
              v50 = (__int64 *)(v48 + 16LL * v49);
              v51 = *v50;
              if ( v13 == *v50 )
                goto LABEL_86;
              v57 = v62;
            }
          }
          v50 = (__int64 *)(v48 + 16LL * v47);
LABEL_86:
          v52 = v50[1];
          v12 = v77;
          v10 = v76;
          if ( (v52 & 1) != 0 )
          {
            v70 = v13;
            v67 = sub_39FAC40(~(-1LL << (v52 >> 58)) & (v52 >> 1));
          }
          else
          {
            v67 = (unsigned int)(*(_DWORD *)(v52 + 16) + 63) >> 6;
            if ( v67 )
            {
              v53 = *(_QWORD **)v52;
              v70 = v13;
              v65 = v76;
              v54 = *(_QWORD *)v52 + 8LL + 8LL * (((unsigned int)(*(_DWORD *)(v52 + 16) + 63) >> 6) - 1);
              v55 = *(_QWORD *)v52 + 8LL;
              v56 = 0;
              while ( 1 )
              {
                v56 += sub_39FAC40(*v53);
                v53 = (_QWORD *)v55;
                if ( v55 == v54 )
                  break;
                v55 += 8;
              }
              v67 = v56;
              v10 = v65;
            }
            else
            {
              v70 = v13;
            }
          }
          goto LABEL_9;
        }
        v29 = *(unsigned int *)(a1 + 32152);
        v30 = *(_QWORD *)(a1 + 32136);
        if ( !(_DWORD)v29 )
          goto LABEL_75;
        v31 = (v29 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v32 = (__int64 *)(v30 + 16LL * v31);
        v33 = *v32;
        if ( v13 != *v32 )
        {
          v43 = 1;
          while ( v33 != -8 )
          {
            v61 = v43 + 1;
            v31 = (v29 - 1) & (v43 + v31);
            v32 = (__int64 *)(v30 + 16LL * v31);
            v33 = *v32;
            if ( v13 == *v32 )
              goto LABEL_40;
            v43 = v61;
          }
LABEL_75:
          v32 = (__int64 *)(v30 + 16 * v29);
        }
LABEL_40:
        v34 = v32[1];
        v12 = v77;
        v10 = v76;
        if ( (v34 & 1) != 0 )
        {
          v35 = sub_39FAC40((v34 >> 1) & ~(-1LL << (v34 >> 58)));
          goto LABEL_42;
        }
        v39 = (unsigned int)(*(_DWORD *)(v34 + 16) + 63) >> 6;
        if ( v39 )
        {
          v40 = *(_QWORD **)v34;
          v64 = v13;
          v35 = 0;
          v63 = v76;
          v41 = v40 + 1;
          v42 = (__int64)&v40[v39];
          while ( 1 )
          {
            v35 += sub_39FAC40(*v40);
            v40 = v41;
            if ( v41 == (_QWORD *)v42 )
              break;
            ++v41;
          }
          v13 = v64;
          v10 = v63;
LABEL_42:
          if ( v35 > v67 )
          {
            v67 = v35;
            v70 = v13;
          }
        }
LABEL_9:
        if ( v72 == ++v11 )
          goto LABEL_17;
      }
      v12 = v77;
      v10 = v76;
      ++v11;
    }
    while ( v72 != v11 );
LABEL_17:
    v2 = v12;
    v16 = v70;
    v1 = v66;
    v3 = v10;
    if ( v70 )
    {
      if ( v10 != v2 )
        goto LABEL_19;
      v44 = &v10[HIDWORD(v78)];
      if ( v44 == v10 )
        goto LABEL_114;
      v45 = v10;
      v46 = 0;
      do
      {
        if ( v70 == *v45 )
          goto LABEL_20;
        if ( *v45 == -2 )
          v46 = v45;
        ++v45;
      }
      while ( v44 != v45 );
      if ( !v46 )
      {
LABEL_114:
        if ( HIDWORD(v78) >= (unsigned int)v78 )
        {
LABEL_19:
          sub_16CCBA0((__int64)&v75, v70);
          v2 = v77;
          v3 = v76;
        }
        else
        {
          ++HIDWORD(v78);
          *v44 = v70;
          v3 = v76;
          ++v75;
          v2 = v77;
        }
      }
      else
      {
        *v46 = v70;
        v2 = v77;
        --v79;
        v3 = v76;
        ++v75;
      }
LABEL_20:
      v4 = 0;
      v71 = 0;
      v73 = 0;
      v68 = *(unsigned int *)(a1 + 376);
      if ( !*(_DWORD *)(a1 + 376) )
        continue;
LABEL_21:
      v17 = *(_QWORD *)(a1 + 368) + v71;
      v18 = *(_QWORD **)(v17 + 1928);
      v19 = *(_QWORD **)(v17 + 1920);
      if ( v18 == v19 )
      {
        v20 = &v19[*(unsigned int *)(v17 + 1940)];
        if ( v19 == v20 )
        {
          v58 = *(_QWORD **)(v17 + 1920);
        }
        else
        {
          do
          {
            if ( v16 == *v19 )
              break;
            ++v19;
          }
          while ( v20 != v19 );
          v58 = v20;
        }
      }
      else
      {
        v20 = &v18[*(unsigned int *)(v17 + 1936)];
        v19 = sub_16CC9F0(v17 + 1912, v16);
        if ( v16 == *v19 )
        {
          v37 = *(_QWORD *)(v17 + 1928);
          if ( v37 == *(_QWORD *)(v17 + 1920) )
            v38 = *(unsigned int *)(v17 + 1940);
          else
            v38 = *(unsigned int *)(v17 + 1936);
          v58 = (_QWORD *)(v37 + 8 * v38);
        }
        else
        {
          v21 = *(_QWORD *)(v17 + 1928);
          if ( v21 != *(_QWORD *)(v17 + 1920) )
          {
            v19 = (_QWORD *)(v21 + 8LL * *(unsigned int *)(v17 + 1936));
            goto LABEL_25;
          }
          v19 = (_QWORD *)(v21 + 8LL * *(unsigned int *)(v17 + 1940));
          v58 = v19;
        }
      }
      for ( ; v58 != v19; ++v19 )
      {
        if ( *v19 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
LABEL_25:
      if ( v19 != v20 )
      {
        v22 = *(unsigned int *)(v17 + 752);
        v23 = 0;
        v24 = 0;
        if ( *(_DWORD *)(v17 + 752) )
        {
          do
          {
            while ( 1 )
            {
              v74 = v16;
              v25 = *(_QWORD *)(v17 + 744) + 96 * v23;
              if ( v16 == *(_QWORD *)(v25 + 80) )
                break;
              v26 = *(_QWORD **)(v25 + 32);
              v27 = &v26[*(unsigned int *)(v25 + 40)];
              if ( v27 != sub_1993010(v26, (__int64)v27, v66) )
                break;
              --v22;
              sub_1994A60(v17, v28);
              v24 = 1;
              if ( v22 == v23 )
                goto LABEL_32;
            }
            ++v23;
          }
          while ( v22 != v23 );
LABEL_32:
          if ( v24 )
            sub_1996C50(v17, v73, a1 + 32128);
        }
      }
      ++v73;
      v71 += 1984;
      if ( v73 == v68 )
      {
        v2 = v77;
        v3 = v76;
        v4 = *(unsigned int *)(a1 + 376);
        continue;
      }
      goto LABEL_21;
    }
    break;
  }
LABEL_94:
  if ( v2 != v3 )
    _libc_free((unsigned __int64)v2);
}
