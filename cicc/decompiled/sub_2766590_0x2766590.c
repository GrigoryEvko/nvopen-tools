// Function: sub_2766590
// Address: 0x2766590
//
__int64 __fastcall sub_2766590(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  _BYTE **v7; // rax
  _BYTE *v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 *v10; // rbx
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdi
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rdi
  int v17; // edx
  int v18; // r8d
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // r9
  _QWORD *v22; // r12
  __int64 v23; // r8
  __int64 v24; // r9
  char **v25; // rax
  char *v26; // r15
  __int64 v27; // r14
  unsigned __int8 v28; // al
  __int64 v29; // rax
  int v30; // ecx
  __int64 v31; // rsi
  int v32; // ecx
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // rdi
  _QWORD *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r14
  __int64 *v42; // r14
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 *v47; // rax
  char v48; // dl
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 *v51; // rdx
  unsigned __int64 v52; // rcx
  __int64 *v53; // rax
  __int64 *v54; // rcx
  __int64 v55; // rsi
  __int64 v56; // rcx
  __int64 *v57; // rax
  unsigned int v58; // esi
  __int64 *v59; // rdi
  __int64 v60; // rcx
  __int64 *v61; // rdx
  char *v62; // r14
  __int64 v63; // rax
  char **v64; // rax
  char v65; // al
  char v66; // dl
  int v67; // eax
  int v68; // edx
  int v69; // r10d
  __int64 v70; // [rsp+18h] [rbp-148h]
  __int64 *v71; // [rsp+28h] [rbp-138h]
  __int64 v72; // [rsp+28h] [rbp-138h]
  __int64 v73; // [rsp+28h] [rbp-138h]
  __m128i v74; // [rsp+30h] [rbp-130h] BYREF
  unsigned __int64 v75[2]; // [rsp+40h] [rbp-120h] BYREF
  unsigned __int64 v76; // [rsp+50h] [rbp-110h]
  __int64 v77; // [rsp+58h] [rbp-108h]
  __int64 v78; // [rsp+60h] [rbp-100h]
  unsigned __int64 *v79; // [rsp+68h] [rbp-F8h]
  __int64 v80; // [rsp+70h] [rbp-F0h]
  __int64 v81; // [rsp+78h] [rbp-E8h]
  __int64 v82; // [rsp+80h] [rbp-E0h]
  __int64 *v83; // [rsp+88h] [rbp-D8h]
  __int64 v84; // [rsp+90h] [rbp-D0h] BYREF
  __int64 *v85; // [rsp+98h] [rbp-C8h]
  __int64 v86; // [rsp+A0h] [rbp-C0h]
  int v87; // [rsp+A8h] [rbp-B8h]
  char v88; // [rsp+ACh] [rbp-B4h]
  _BYTE *v89; // [rsp+B0h] [rbp-B0h] BYREF

  v2 = 0;
  v78 = 0;
  v79 = 0;
  v82 = 0;
  v83 = 0;
  v75[1] = 8;
  v75[0] = sub_22077B0(0x40u);
  v4 = (__int64 *)(v75[0] + 24);
  v5 = sub_22077B0(0x200u);
  *(_DWORD *)(a1 + 32) = 0;
  *v4 = v5;
  v6 = v5 + 512;
  v77 = v5;
  v81 = v5;
  v76 = v5;
  v80 = v5;
  v85 = (__int64 *)&v89;
  v7 = *(_BYTE ***)(a2 - 8);
  v79 = (unsigned __int64 *)v4;
  v78 = v6;
  v83 = v4;
  v82 = v6;
  v84 = 0;
  v86 = 16;
  v87 = 0;
  v88 = 1;
  v8 = *v7;
  if ( *v8 != 84 )
    goto LABEL_2;
  v14 = *(_QWORD *)(a1 + 8);
  v15 = *(_QWORD *)(a2 + 40);
  v16 = *(_QWORD *)(v14 + 8);
  v17 = *(_DWORD *)(v14 + 24);
  if ( !v17 )
    goto LABEL_2;
  v18 = v17 - 1;
  v19 = (v17 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v20 = (__int64 *)(v16 + 16LL * v19);
  v21 = *v20;
  if ( v15 != *v20 )
  {
    v68 = 1;
    while ( v21 != -4096 )
    {
      v69 = v68 + 1;
      v19 = v18 & (v68 + v19);
      v20 = (__int64 *)(v16 + 16LL * v19);
      v21 = *v20;
      if ( v15 == *v20 )
        goto LABEL_10;
      v68 = v69;
    }
LABEL_88:
    v2 = 0;
    goto LABEL_2;
  }
LABEL_10:
  v22 = (_QWORD *)v20[1];
  if ( !v22 )
    goto LABEL_88;
  v89 = v8;
  v74 = (__m128i)(unsigned __int64)v8;
  HIDWORD(v86) = 1;
  v84 = 1;
  sub_2765810(v75, &v74);
  v25 = (char **)v76;
  if ( v80 == v76 )
  {
LABEL_87:
    v2 = 1;
    goto LABEL_27;
  }
  while ( 1 )
  {
    v26 = *v25;
    v27 = (__int64)v25[1];
    if ( v25 == (char **)(v78 - 16) )
    {
      j_j___libc_free_0(v77);
      v37 = *++v79 + 512;
      v77 = *v79;
      v78 = v37;
      v76 = v77;
    }
    else
    {
      v76 += 16LL;
    }
    v28 = *v26;
    if ( (unsigned __int8)*v26 <= 0x1Cu )
    {
      if ( v28 <= 0x15u )
        goto LABEL_14;
      goto LABEL_20;
    }
    if ( v28 == 84 )
    {
      v38 = *((_QWORD *)v26 - 1);
      v39 = *((_DWORD *)v26 + 1) & 0x7FFFFFF;
      v40 = 32LL * *((unsigned int *)v26 + 18);
      v24 = v38 + v40;
      v41 = v40 + 8 * v39;
      v71 = (__int64 *)(v38 + v41);
      if ( v38 + v41 == v38 + v40 )
        goto LABEL_14;
      v42 = (__int64 *)(v38 + v40);
      while ( 1 )
      {
        v43 = *v42;
        v44 = 0x1FFFFFFFE0LL;
        if ( (_DWORD)v39 )
        {
          v45 = 0;
          do
          {
            if ( v43 == *(_QWORD *)(v38 + 32LL * *((unsigned int *)v26 + 18) + 8 * v45) )
            {
              v44 = 32 * v45;
              goto LABEL_37;
            }
            ++v45;
          }
          while ( (_DWORD)v39 != (_DWORD)v45 );
          v44 = 0x1FFFFFFFE0LL;
        }
LABEL_37:
        v46 = *(_QWORD *)(v38 + v44);
        if ( !v88 )
          goto LABEL_44;
        v47 = v85;
        v39 = (__int64)&v85[HIDWORD(v86)];
        if ( v85 == (__int64 *)v39 )
          break;
        while ( v46 != *v47 )
        {
          if ( (__int64 *)v39 == ++v47 )
            goto LABEL_47;
        }
LABEL_42:
        if ( v71 == ++v42 )
          goto LABEL_14;
        v38 = *((_QWORD *)v26 - 1);
        v39 = *((_DWORD *)v26 + 1) & 0x7FFFFFF;
      }
LABEL_47:
      if ( HIDWORD(v86) >= (unsigned int)v86 )
      {
LABEL_44:
        v70 = *v42;
        sub_C8CC70((__int64)&v84, v46, v39, v43, v23, v24);
        v43 = v70;
        if ( !v48 )
          goto LABEL_42;
      }
      else
      {
        ++HIDWORD(v86);
        *(_QWORD *)v39 = v46;
        ++v84;
      }
      v74.m128i_i64[0] = v46;
      v74.m128i_i64[1] = v43;
      sub_2765810(v75, &v74);
      goto LABEL_42;
    }
    if ( v28 == 86 )
      break;
LABEL_20:
    if ( byte_4FFAFC8 )
    {
      v29 = *(_QWORD *)(a1 + 8);
      v30 = *(_DWORD *)(v29 + 24);
      v31 = *(_QWORD *)(v29 + 8);
      if ( v30 )
      {
        v32 = v30 - 1;
        v33 = v32 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v34 = (__int64 *)(v31 + 16LL * v33);
        v35 = *v34;
        if ( v27 == *v34 )
        {
LABEL_23:
          v36 = (_QWORD *)v34[1];
          if ( v22 == v36 )
            goto LABEL_26;
          while ( v36 )
          {
            v36 = (_QWORD *)*v36;
            if ( v22 == v36 )
              goto LABEL_26;
          }
        }
        else
        {
          v67 = 1;
          while ( v35 != -4096 )
          {
            v23 = (unsigned int)(v67 + 1);
            v33 = v32 & (v67 + v33);
            v34 = (__int64 *)(v31 + 16LL * v33);
            v35 = *v34;
            if ( v27 == *v34 )
              goto LABEL_23;
            v67 = v23;
          }
        }
      }
    }
LABEL_14:
    v25 = (char **)v76;
    if ( v80 == v76 )
      goto LABEL_87;
  }
  v49 = *((_QWORD *)v26 + 2);
  if ( !v49 || *(_QWORD *)(v49 + 8) )
    goto LABEL_26;
  v50 = *(_QWORD *)(v49 + 24);
  if ( !v50 )
    BUG();
  v51 = (__int64 *)*((_QWORD *)v26 + 5);
  v52 = v51[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (__int64 *)v52 == v51 + 6 || !v52 || (unsigned int)*(unsigned __int8 *)(v52 - 24) - 30 > 0xA )
    BUG();
  if ( *(_BYTE *)(v52 - 24) != 31
    || (*(_DWORD *)(v52 - 20) & 0x7FFFFFF) != 1
    || *(_BYTE *)v50 == 84
    && v51 != *(__int64 **)(*(_QWORD *)(v50 - 8)
                          + 32LL * *(unsigned int *)(v50 + 72)
                          + 8LL * (unsigned int)((v49 - *(_QWORD *)(v50 - 8)) >> 5)) )
  {
    goto LABEL_26;
  }
  v53 = *(__int64 **)(a1 + 24);
  v54 = &v53[2 * *(unsigned int *)(a1 + 32)];
  if ( v53 == v54 )
  {
LABEL_64:
    v56 = *((_QWORD *)v26 - 8);
    if ( !v88 )
      goto LABEL_78;
    v57 = v85;
    v58 = HIDWORD(v86);
    v59 = &v85[HIDWORD(v86)];
    v51 = v85;
    if ( v85 != v59 )
    {
      while ( v56 != *v51 )
      {
        if ( v59 == ++v51 )
          goto LABEL_84;
      }
      v60 = *((_QWORD *)v26 - 4);
      goto LABEL_70;
    }
LABEL_84:
    if ( HIDWORD(v86) < (unsigned int)v86 )
    {
      ++HIDWORD(v86);
      *v59 = v56;
      ++v84;
    }
    else
    {
LABEL_78:
      v72 = *((_QWORD *)v26 - 8);
      sub_C8CC70((__int64)&v84, v72, (__int64)v51, v56, v23, v24);
      v65 = v88;
      v56 = v72;
      if ( !(_BYTE)v61 )
      {
LABEL_79:
        v60 = *((_QWORD *)v26 - 4);
        if ( !v65 )
          goto LABEL_80;
        v57 = v85;
        v58 = HIDWORD(v86);
LABEL_70:
        v61 = &v57[v58];
        if ( v57 != v61 )
        {
          while ( v60 != *v57 )
          {
            if ( v61 == ++v57 )
              goto LABEL_82;
          }
          goto LABEL_74;
        }
LABEL_82:
        if ( v58 < (unsigned int)v86 )
        {
          HIDWORD(v86) = v58 + 1;
          *v61 = v60;
          ++v84;
LABEL_81:
          v74.m128i_i64[0] = v60;
          v74.m128i_i64[1] = v27;
          sub_2765810(v75, &v74);
LABEL_74:
          v62 = *(char **)(*((_QWORD *)v26 + 2) + 24LL);
          if ( *v62 == 84 )
          {
            v63 = *(unsigned int *)(a1 + 32);
            if ( v63 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 36) )
            {
              sub_C8D5F0(a1 + 24, (const void *)(a1 + 40), v63 + 1, 0x10u, v23, v24);
              v63 = *(unsigned int *)(a1 + 32);
            }
            v64 = (char **)(*(_QWORD *)(a1 + 24) + 16 * v63);
            *v64 = v26;
            v64[1] = v62;
            ++*(_DWORD *)(a1 + 32);
          }
          goto LABEL_14;
        }
LABEL_80:
        v73 = v60;
        sub_C8CC70((__int64)&v84, v60, (__int64)v61, v60, v23, v24);
        v60 = v73;
        if ( !v66 )
          goto LABEL_74;
        goto LABEL_81;
      }
    }
    v74.m128i_i64[0] = v56;
    v74.m128i_i64[1] = v27;
    sub_2765810(v75, &v74);
    v65 = v88;
    goto LABEL_79;
  }
  while ( 1 )
  {
    v55 = *v53;
    if ( v26 != *(char **)(*v53 - 64) && v26 != *(char **)(v55 - 32) && v51 == *(__int64 **)(v55 + 40) )
      break;
    v53 += 2;
    if ( v54 == v53 )
      goto LABEL_64;
  }
LABEL_26:
  v2 = 0;
LABEL_27:
  if ( !v88 )
    _libc_free((unsigned __int64)v85);
LABEL_2:
  v9 = v75[0];
  if ( v75[0] )
  {
    v10 = v79;
    v11 = (unsigned __int64)(v83 + 1);
    if ( v83 + 1 > (__int64 *)v79 )
    {
      do
      {
        v12 = *v10++;
        j_j___libc_free_0(v12);
      }
      while ( v11 > (unsigned __int64)v10 );
      v9 = v75[0];
    }
    j_j___libc_free_0(v9);
  }
  return v2;
}
