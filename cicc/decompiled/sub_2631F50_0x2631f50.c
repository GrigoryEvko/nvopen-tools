// Function: sub_2631F50
// Address: 0x2631f50
//
__int64 __fastcall sub_2631F50(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 v11; // r14
  unsigned __int8 v12; // dl
  __int64 v13; // rcx
  __int64 v14; // rdi
  unsigned int v15; // esi
  __int64 v16; // rdx
  __int64 **v17; // rdi
  __int64 **v18; // r14
  __int64 **v19; // rbx
  __int64 *v20; // rdi
  __int64 **v22; // rax
  __int64 v23; // rdx
  __int64 **v24; // rsi
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 **v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // rbx
  __int64 *v30; // r14
  unsigned int v31; // eax
  __int64 *v32; // rcx
  __int64 v33; // rdi
  int v34; // esi
  __int64 *v35; // r10
  int v36; // edx
  int v37; // esi
  unsigned int v38; // r10d
  __int64 **v39; // rdx
  __int64 *v40; // rcx
  int v41; // eax
  __int64 v42; // rax
  __int64 v43; // rbx
  unsigned __int64 v44; // rdx
  int v45; // r11d
  __int64 v46; // [rsp+8h] [rbp-B8h]
  __int64 v47; // [rsp+10h] [rbp-B0h]
  unsigned int v48; // [rsp+18h] [rbp-A8h]
  int v49; // [rsp+18h] [rbp-A8h]
  __int64 v50; // [rsp+18h] [rbp-A8h]
  __int64 v51; // [rsp+18h] [rbp-A8h]
  __int64 v52; // [rsp+18h] [rbp-A8h]
  __int64 v54; // [rsp+28h] [rbp-98h]
  __int64 *v55; // [rsp+30h] [rbp-90h] BYREF
  __int64 *v56; // [rsp+38h] [rbp-88h] BYREF
  __int64 v57; // [rsp+40h] [rbp-80h] BYREF
  __int64 v58; // [rsp+48h] [rbp-78h]
  __int64 v59; // [rsp+50h] [rbp-70h]
  __int64 v60; // [rsp+58h] [rbp-68h]
  __int64 **v61; // [rsp+60h] [rbp-60h] BYREF
  __int64 v62; // [rsp+68h] [rbp-58h]
  _BYTE v63[80]; // [rsp+70h] [rbp-50h] BYREF

  v62 = 0x400000000LL;
  v9 = *(_QWORD *)(a2 + 16);
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = (__int64 **)v63;
  v54 = a3 + 16;
  if ( !v9 )
    return sub_C7D6A0(v58, 8LL * (unsigned int)v60, 8);
  while ( 1 )
  {
    v10 = *(__int64 **)(v9 + 24);
    v11 = *(_QWORD *)(v9 + 8);
    v12 = *(_BYTE *)v10;
    if ( *(_BYTE *)v10 == 4
      || v12 == 7
      || v12 == 85 && (__int64 *)v9 == v10 - 4 && ((*(_BYTE *)(a2 + 33) & 0x40) != 0 || !a4) )
    {
      goto LABEL_18;
    }
    v13 = *(unsigned int *)(a1 + 224);
    v14 = *(_QWORD *)(a1 + 208);
    if ( (_DWORD)v13 )
    {
      v15 = (v13 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      a5 = v14 + 8LL * v15;
      a6 = *(_QWORD *)a5;
      if ( v10 == *(__int64 **)a5 )
      {
LABEL_9:
        if ( a5 != v14 + 8 * v13 )
          goto LABEL_18;
      }
      else
      {
        a5 = 1;
        while ( a6 != -4096 )
        {
          v15 = (v13 - 1) & (a5 + v15);
          v48 = a5 + 1;
          a5 = v14 + 8LL * v15;
          a6 = *(_QWORD *)a5;
          if ( v10 == *(__int64 **)a5 )
            goto LABEL_9;
          a5 = v48;
        }
      }
    }
    if ( v12 > 0x15u )
      break;
    v55 = *(__int64 **)(v9 + 24);
    if ( *(_BYTE *)v10 <= 3u )
      break;
    if ( (_DWORD)v59 )
    {
      v37 = v60;
      if ( (_DWORD)v60 )
      {
        a5 = (unsigned int)(v60 - 1);
        v38 = a5 & (((unsigned int)v10 >> 4) ^ ((unsigned int)v10 >> 9));
        v39 = (__int64 **)(v58 + 8LL * v38);
        v40 = *v39;
        if ( v10 == *v39 )
          goto LABEL_18;
        v49 = 1;
        a6 = 0;
        while ( v40 != (__int64 *)-4096LL )
        {
          if ( !a6 && v40 == (__int64 *)-8192LL )
            a6 = (__int64)v39;
          v38 = a5 & (v49 + v38);
          v39 = (__int64 **)(v58 + 8LL * v38);
          v40 = *v39;
          if ( v10 == *v39 )
            goto LABEL_18;
          ++v49;
        }
        if ( !a6 )
          a6 = (__int64)v39;
        ++v57;
        v41 = v59 + 1;
        v56 = (__int64 *)a6;
        if ( 4 * ((int)v59 + 1) < (unsigned int)(3 * v60) )
        {
          if ( (int)v60 - HIDWORD(v59) - v41 > (unsigned int)v60 >> 3 )
          {
LABEL_68:
            LODWORD(v59) = v41;
            if ( *(_QWORD *)a6 != -4096 )
              --HIDWORD(v59);
            *(_QWORD *)a6 = v10;
            v42 = (unsigned int)v62;
            v43 = (__int64)v55;
            v44 = (unsigned int)v62 + 1LL;
            if ( v44 > HIDWORD(v62) )
            {
              v52 = a1;
              sub_C8D5F0((__int64)&v61, v63, v44, 8u, a5, a6);
              v42 = (unsigned int)v62;
              a1 = v52;
            }
            v61[v42] = (__int64 *)v43;
            LODWORD(v62) = v62 + 1;
            goto LABEL_18;
          }
          v51 = a1;
LABEL_96:
          sub_2631D80((__int64)&v57, v37);
          sub_262CED0((__int64)&v57, (__int64 *)&v55, &v56);
          v10 = v55;
          a6 = (__int64)v56;
          a1 = v51;
          v41 = v59 + 1;
          goto LABEL_68;
        }
      }
      else
      {
        ++v57;
        v56 = 0;
      }
      v51 = a1;
      v37 = 2 * v60;
      goto LABEL_96;
    }
    v22 = v61;
    v23 = 8LL * (unsigned int)v62;
    v24 = &v61[(unsigned __int64)v23 / 8];
    v25 = v23 >> 3;
    v26 = v23 >> 5;
    if ( v26 )
    {
      v27 = &v61[4 * v26];
      while ( v10 != *v22 )
      {
        if ( v10 == v22[1] )
        {
          ++v22;
          goto LABEL_42;
        }
        if ( v10 == v22[2] )
        {
          v22 += 2;
          goto LABEL_42;
        }
        if ( v10 == v22[3] )
        {
          v22 += 3;
          goto LABEL_42;
        }
        v22 += 4;
        if ( v27 == v22 )
        {
          v25 = v24 - v22;
          goto LABEL_74;
        }
      }
      goto LABEL_42;
    }
LABEL_74:
    if ( v25 == 2 )
      goto LABEL_82;
    if ( v25 != 3 )
    {
      if ( v25 != 1 )
        goto LABEL_43;
LABEL_77:
      if ( v10 != *v22 )
        goto LABEL_43;
      goto LABEL_42;
    }
    if ( v10 != *v22 )
    {
      ++v22;
LABEL_82:
      if ( v10 != *v22 )
      {
        ++v22;
        goto LABEL_77;
      }
    }
LABEL_42:
    if ( v24 != v22 )
      goto LABEL_18;
LABEL_43:
    if ( (unsigned __int64)(unsigned int)v62 + 1 > HIDWORD(v62) )
    {
      v50 = a1;
      sub_C8D5F0((__int64)&v61, v63, (unsigned int)v62 + 1LL, 8u, a5, a6);
      a1 = v50;
      v24 = &v61[(unsigned int)v62];
    }
    *v24 = v10;
    v28 = (unsigned int)(v62 + 1);
    LODWORD(v62) = v28;
    if ( (unsigned int)v28 > 4 )
    {
      v47 = v11;
      v46 = a1;
      v29 = (__int64 *)v61;
      v30 = (__int64 *)&v61[v28];
      while ( 1 )
      {
        v34 = v60;
        if ( !(_DWORD)v60 )
          break;
        a6 = (unsigned int)(v60 - 1);
        a5 = v58;
        v31 = a6 & (((unsigned int)*v29 >> 9) ^ ((unsigned int)*v29 >> 4));
        v32 = (__int64 *)(v58 + 8LL * v31);
        v33 = *v32;
        if ( *v29 != *v32 )
        {
          v45 = 1;
          v35 = 0;
          while ( v33 != -4096 )
          {
            if ( !v35 && v33 == -8192 )
              v35 = v32;
            v31 = a6 & (v45 + v31);
            v32 = (__int64 *)(v58 + 8LL * v31);
            v33 = *v32;
            if ( *v29 == *v32 )
              goto LABEL_48;
            ++v45;
          }
          if ( !v35 )
            v35 = v32;
          ++v57;
          v36 = v59 + 1;
          v56 = v35;
          if ( 4 * ((int)v59 + 1) < (unsigned int)(3 * v60) )
          {
            if ( (int)v60 - HIDWORD(v59) - v36 <= (unsigned int)v60 >> 3 )
            {
LABEL_52:
              sub_2631D80((__int64)&v57, v34);
              sub_262CED0((__int64)&v57, v29, &v56);
              v35 = v56;
              v36 = v59 + 1;
            }
            LODWORD(v59) = v36;
            if ( *v35 != -4096 )
              --HIDWORD(v59);
            *v35 = *v29;
            goto LABEL_48;
          }
LABEL_51:
          v34 = 2 * v60;
          goto LABEL_52;
        }
LABEL_48:
        if ( v30 == ++v29 )
        {
          v11 = v47;
          a1 = v46;
          goto LABEL_18;
        }
      }
      ++v57;
      v56 = 0;
      goto LABEL_51;
    }
LABEL_18:
    if ( !v11 )
      goto LABEL_19;
LABEL_3:
    v9 = v11;
  }
  if ( !*(_QWORD *)v9 )
  {
    *(_QWORD *)v9 = a3;
    if ( !a3 )
      goto LABEL_18;
    goto LABEL_15;
  }
  **(_QWORD **)(v9 + 16) = v11;
  if ( v11 )
  {
    *(_QWORD *)(v11 + 16) = *(_QWORD *)(v9 + 16);
    *(_QWORD *)v9 = a3;
    if ( !a3 )
      goto LABEL_3;
    goto LABEL_15;
  }
  *(_QWORD *)v9 = a3;
  if ( a3 )
  {
LABEL_15:
    v16 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v9 + 8) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = v9 + 8;
    *(_QWORD *)(v9 + 16) = v54;
    *(_QWORD *)(a3 + 16) = v9;
    goto LABEL_18;
  }
LABEL_19:
  v17 = v61;
  v18 = &v61[(unsigned int)v62];
  if ( v18 != v61 )
  {
    v19 = v61;
    do
    {
      v20 = *v19++;
      sub_ADBE50(v20, (__int64 *)a2, (_BYTE *)a3);
    }
    while ( v18 != v19 );
    v17 = v61;
  }
  if ( v17 != (__int64 **)v63 )
    _libc_free((unsigned __int64)v17);
  return sub_C7D6A0(v58, 8LL * (unsigned int)v60, 8);
}
