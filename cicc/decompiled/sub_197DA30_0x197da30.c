// Function: sub_197DA30
// Address: 0x197da30
//
__int64 __fastcall sub_197DA30(__int64 a1, _QWORD *a2, __m128i a3, __m128i a4)
{
  _QWORD *v4; // rbx
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r10d
  _QWORD *v9; // rax
  unsigned int v10; // r8d
  _QWORD *v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r14
  _QWORD *v15; // r13
  unsigned int v16; // esi
  int v17; // edx
  __int64 v18; // r9
  int v19; // r11d
  _QWORD *v20; // r10
  _QWORD *v21; // r13
  _QWORD *v22; // rbx
  __int64 v23; // r9
  __int64 v24; // rcx
  unsigned int v25; // edx
  _QWORD *v26; // rax
  __int64 v27; // r10
  _QWORD *v28; // rdi
  __int64 v29; // rsi
  unsigned int v30; // eax
  int v31; // edx
  _QWORD *v32; // r8
  __int64 v33; // r9
  __int64 v34; // rax
  unsigned int v36; // esi
  __int64 v37; // rdi
  unsigned int v38; // r9d
  __int64 v39; // r8
  unsigned int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // r11
  unsigned int v43; // r10d
  __int64 v44; // r8
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // r14
  unsigned int v48; // eax
  int v49; // r11d
  __int64 v50; // rsi
  _QWORD *v51; // r10
  int v52; // r11d
  unsigned int v53; // eax
  __int64 v54; // r9
  int v55; // r10d
  unsigned int v56; // r15d
  _QWORD *v57; // r9
  __int64 v58; // rdi
  int v59; // r11d
  int v60; // eax
  int v61; // eax
  int v62; // ecx
  __int64 v63; // rax
  int v64; // ecx
  _QWORD *v65; // [rsp+10h] [rbp-60h]
  __int64 v67; // [rsp+20h] [rbp-50h] BYREF
  __int64 v68; // [rsp+28h] [rbp-48h]
  __int64 v69; // [rsp+30h] [rbp-40h]
  unsigned int v70; // [rsp+38h] [rbp-38h]

  v4 = (_QWORD *)*a2;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  if ( !v4 )
  {
    v7 = 0;
    return j___libc_free_0(v7);
  }
  v6 = 0;
  v7 = 0;
  v65 = a2;
  while ( 1 )
  {
    v14 = v4[1];
    v15 = v4 + 1;
    if ( !v6 )
    {
      ++v67;
      goto LABEL_12;
    }
    v8 = 1;
    v9 = 0;
    v10 = (v6 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v11 = (_QWORD *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( v14 != *v11 )
    {
      while ( v12 != -8 )
      {
        if ( !v9 && v12 == -16 )
          v9 = v11;
        v10 = (v6 - 1) & (v8 + v10);
        v11 = (_QWORD *)(v7 + 16LL * v10);
        v12 = *v11;
        if ( v14 == *v11 )
          goto LABEL_4;
        ++v8;
      }
      if ( !v9 )
        v9 = v11;
      ++v67;
      v17 = v69 + 1;
      if ( 4 * ((int)v69 + 1) < 3 * v6 )
      {
        if ( v6 - (v17 + HIDWORD(v69)) <= v6 >> 3 )
        {
          sub_197D390((__int64)&v67, v6);
          if ( !v70 )
          {
LABEL_113:
            LODWORD(v69) = v69 + 1;
            BUG();
          }
          v55 = 1;
          v56 = (v70 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v57 = 0;
          v17 = v69 + 1;
          v9 = (_QWORD *)(v68 + 16LL * v56);
          v58 = *v9;
          if ( v14 != *v9 )
          {
            while ( v58 != -8 )
            {
              if ( v58 == -16 && !v57 )
                v57 = v9;
              v56 = (v70 - 1) & (v55 + v56);
              v9 = (_QWORD *)(v68 + 16LL * v56);
              v58 = *v9;
              if ( v14 == *v9 )
                goto LABEL_29;
              ++v55;
            }
            if ( v57 )
              v9 = v57;
          }
        }
        goto LABEL_29;
      }
LABEL_12:
      sub_197D390((__int64)&v67, 2 * v6);
      if ( !v70 )
        goto LABEL_113;
      v16 = (v70 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v17 = v69 + 1;
      v9 = (_QWORD *)(v68 + 16LL * v16);
      v18 = *v9;
      if ( v14 != *v9 )
      {
        v19 = 1;
        v20 = 0;
        while ( v18 != -8 )
        {
          if ( !v20 && v18 == -16 )
            v20 = v9;
          v16 = (v70 - 1) & (v19 + v16);
          v9 = (_QWORD *)(v68 + 16LL * v16);
          v18 = *v9;
          if ( v14 == *v9 )
            goto LABEL_29;
          ++v19;
        }
        if ( v20 )
          v9 = v20;
      }
LABEL_29:
      LODWORD(v69) = v17;
      if ( *v9 != -8 )
        --HIDWORD(v69);
      *v9 = v14;
      v9[1] = v15;
      goto LABEL_7;
    }
LABEL_4:
    v13 = v11[1];
    if ( !v13 )
      goto LABEL_8;
    if ( *(_QWORD *)(*(_QWORD *)(v13 + 8) + 40LL) == *(_QWORD *)(v4[2] + 40LL)
      && sub_197D610(v4 + 1, a1 + 64, *(_QWORD *)a1, a3, a4)
      && sub_197D610((_QWORD *)v11[1], a1 + 64, *(_QWORD *)a1, a3, a4) )
    {
      v36 = *(_DWORD *)(a1 + 32);
      v37 = *(_QWORD *)(a1 + 16);
      if ( v36 )
      {
        v38 = v36 - 1;
        v39 = *(_QWORD *)(v11[1] + 8LL);
        v40 = (v36 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
        v41 = (__int64 *)(v37 + 16LL * v40);
        v42 = *v41;
        if ( v39 == *v41 )
        {
LABEL_51:
          v43 = *((_DWORD *)v41 + 2);
          v44 = v4[2];
        }
        else
        {
          v61 = 1;
          while ( v42 != -8 )
          {
            v64 = v61 + 1;
            v40 = v38 & (v61 + v40);
            v41 = (__int64 *)(v37 + 16LL * v40);
            v42 = *v41;
            if ( v39 == *v41 )
              goto LABEL_51;
            v61 = v64;
          }
          v44 = v4[2];
          v43 = *(_DWORD *)(v37 + 16LL * v36 + 8);
        }
        v45 = v38 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
        v46 = (__int64 *)(v37 + 16LL * v45);
        v47 = *v46;
        if ( v44 == *v46 )
        {
LABEL_53:
          v48 = *((_DWORD *)v46 + 2);
        }
        else
        {
          v60 = 1;
          while ( v47 != -8 )
          {
            v62 = v60 + 1;
            v63 = v38 & (v45 + v60);
            v45 = v63;
            v46 = (__int64 *)(v37 + 16 * v63);
            v47 = *v46;
            if ( v44 == *v46 )
              goto LABEL_53;
            v60 = v62;
          }
          v48 = *(_DWORD *)(v37 + 16LL * v36 + 8);
        }
        if ( v43 < v48 )
          v11[1] = v15;
      }
    }
    else
    {
      v11[1] = 0;
    }
LABEL_7:
    v7 = v68;
LABEL_8:
    v4 = (_QWORD *)*v4;
    if ( !v4 )
      break;
    v6 = v70;
  }
  v21 = v65;
  v22 = (_QWORD *)*v65;
  if ( *v65 )
  {
    v23 = v7;
    while ( v70 )
    {
      v24 = v22[1];
      v25 = (v70 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v26 = (_QWORD *)(v23 + 16LL * v25);
      v27 = *v26;
      if ( v24 != *v26 )
      {
        v32 = 0;
        v49 = 1;
        while ( v27 != -8 )
        {
          if ( v27 == -16 && !v32 )
            v32 = v26;
          v25 = (v70 - 1) & (v49 + v25);
          v26 = (_QWORD *)(v23 + 16LL * v25);
          v27 = *v26;
          if ( v24 == *v26 )
            goto LABEL_35;
          ++v49;
        }
        if ( !v32 )
          v32 = v26;
        ++v67;
        v31 = v69 + 1;
        if ( 4 * ((int)v69 + 1) < 3 * v70 )
        {
          if ( v70 - HIDWORD(v69) - v31 <= v70 >> 3 )
          {
            sub_197D390((__int64)&v67, v70);
            if ( !v70 )
            {
LABEL_112:
              LODWORD(v69) = v69 + 1;
              BUG();
            }
            v50 = v22[1];
            v51 = 0;
            v52 = 1;
            v53 = (v70 - 1) & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
            v31 = v69 + 1;
            v32 = (_QWORD *)(v68 + 16LL * v53);
            v54 = *v32;
            if ( *v32 != v50 )
            {
              while ( v54 != -8 )
              {
                if ( v54 == -16 && !v51 )
                  v51 = v32;
                v53 = (v70 - 1) & (v52 + v53);
                v32 = (_QWORD *)(v68 + 16LL * v53);
                v54 = *v32;
                if ( v50 == *v32 )
                  goto LABEL_41;
                ++v52;
              }
LABEL_65:
              if ( v51 )
                v32 = v51;
            }
          }
LABEL_41:
          LODWORD(v69) = v31;
          if ( *v32 != -8 )
            --HIDWORD(v69);
          v34 = v22[1];
          v32[1] = 0;
          *v32 = v34;
          v28 = (_QWORD *)*v21;
          goto LABEL_44;
        }
LABEL_39:
        sub_197D390((__int64)&v67, 2 * v70);
        if ( !v70 )
          goto LABEL_112;
        v29 = v22[1];
        v30 = (v70 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v31 = v69 + 1;
        v32 = (_QWORD *)(v68 + 16LL * v30);
        v33 = *v32;
        if ( v29 != *v32 )
        {
          v51 = 0;
          v59 = 1;
          while ( v33 != -8 )
          {
            if ( v33 == -16 && !v51 )
              v51 = v32;
            v30 = (v70 - 1) & (v59 + v30);
            v32 = (_QWORD *)(v68 + 16LL * v30);
            v33 = *v32;
            if ( v29 == *v32 )
              goto LABEL_41;
            ++v59;
          }
          goto LABEL_65;
        }
        goto LABEL_41;
      }
LABEL_35:
      v28 = (_QWORD *)*v21;
      if ( (_QWORD *)v26[1] == v22 + 1 )
      {
        v21 = (_QWORD *)*v21;
        v22 = (_QWORD *)*v28;
        if ( !*v28 )
          goto LABEL_45;
      }
      else
      {
LABEL_44:
        *v21 = *v28;
        j_j___libc_free_0(v28, 24);
        v22 = (_QWORD *)*v21;
        v23 = v68;
        if ( !*v21 )
        {
LABEL_45:
          v7 = v23;
          return j___libc_free_0(v7);
        }
      }
    }
    ++v67;
    goto LABEL_39;
  }
  return j___libc_free_0(v7);
}
