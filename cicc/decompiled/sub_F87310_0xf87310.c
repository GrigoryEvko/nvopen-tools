// Function: sub_F87310
// Address: 0xf87310
//
__int64 __fastcall sub_F87310(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r8
  _QWORD *v14; // rdx
  __int64 v15; // r8
  unsigned int v16; // edi
  __int64 *v17; // rax
  __int64 v18; // r9
  _QWORD *v19; // rax
  int v21; // eax
  unsigned __int16 v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 *v28; // r12
  __int64 *v29; // rbx
  _BYTE *v30; // r12
  _BYTE *v31; // r13
  _QWORD *v32; // r14
  _QWORD *v33; // rax
  bool v34; // bl
  int v35; // edx
  int v36; // edi
  __int64 v37; // r8
  _QWORD *v38; // rdx
  _QWORD *v39; // r9
  __int64 v40; // rax
  _QWORD *v41; // rax
  int v42; // edx
  __int64 v43; // r8
  unsigned int v44; // edx
  _QWORD *v45; // rbx
  _QWORD *v46; // rdi
  __int64 v47; // rax
  __int64 *v48; // rdx
  __int64 *v49; // rax
  int v50; // eax
  int v51; // r9d
  int v52; // edx
  int v53; // ecx
  int v54; // r9d
  int v55; // r10d
  _QWORD *v56; // [rsp+0h] [rbp-1E0h]
  __int64 v57; // [rsp+20h] [rbp-1C0h]
  _QWORD v58[2]; // [rsp+30h] [rbp-1B0h] BYREF
  _QWORD v59[2]; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 v60; // [rsp+50h] [rbp-190h] BYREF
  __int64 v61; // [rsp+58h] [rbp-188h]
  _QWORD *v62; // [rsp+60h] [rbp-180h]
  __int64 v63; // [rsp+70h] [rbp-170h] BYREF
  __int64 v64; // [rsp+78h] [rbp-168h]
  __int64 v65; // [rsp+80h] [rbp-160h]
  _BYTE *v66; // [rsp+90h] [rbp-150h] BYREF
  __int64 v67; // [rsp+98h] [rbp-148h]
  _BYTE v68[128]; // [rsp+A0h] [rbp-140h] BYREF
  const char *v69; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v70; // [rsp+128h] [rbp-B8h]
  _BYTE v71[176]; // [rsp+130h] [rbp-B0h] BYREF

  v2 = a2;
  if ( *(_BYTE *)a2 <= 0x1Cu || !*(_BYTE *)(a1 + 24) )
    return v2;
  v4 = *(_QWORD *)a1;
  v5 = *(_QWORD *)(a1 + 576);
  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_QWORD *)(v4 + 48);
  v8 = *(_DWORD *)(v7 + 24);
  v9 = *(_QWORD *)(v7 + 8);
  if ( !v8 )
  {
    if ( v5 )
      return v2;
LABEL_67:
    BUG();
  }
  v10 = v8 - 1;
  v11 = v10 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( v6 == *v12 )
  {
LABEL_5:
    v14 = (_QWORD *)v12[1];
  }
  else
  {
    v50 = 1;
    while ( v13 != -4096 )
    {
      v54 = v50 + 1;
      v11 = v10 & (v50 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v6 == *v12 )
        goto LABEL_5;
      v50 = v54;
    }
    v14 = 0;
  }
  if ( !v5 )
    goto LABEL_67;
  v15 = *(_QWORD *)(v5 + 16);
  v16 = v10 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
  v17 = (__int64 *)(v9 + 16LL * v16);
  v18 = *v17;
  if ( v15 == *v17 )
  {
LABEL_8:
    v19 = (_QWORD *)v17[1];
    if ( !v14 || v19 == v14 )
      return v2;
    while ( v19 )
    {
      v19 = (_QWORD *)*v19;
      if ( v19 == v14 )
        return v2;
    }
    goto LABEL_16;
  }
  v21 = 1;
  while ( v18 != -4096 )
  {
    v55 = v21 + 1;
    v16 = v10 & (v21 + v16);
    v17 = (__int64 *)(v9 + 16LL * v16);
    v18 = *v17;
    if ( v15 == *v17 )
      goto LABEL_8;
    v21 = v55;
  }
  if ( v14 )
  {
LABEL_16:
    v22 = *(_WORD *)(a1 + 584);
    if ( *(_BYTE *)(*(_QWORD *)(v2 + 8) + 8LL) == 12 )
    {
      v49 = (__int64 *)sub_BD5C60(v2);
      v24 = sub_BCE3C0(v49, 0);
    }
    else
    {
      v23 = (_QWORD *)sub_BD5C60(v2);
      v24 = sub_BCB2D0(v23);
    }
    v71[17] = 1;
    v69 = "tmp.lcssa.user";
    v71[16] = 3;
    v57 = sub_B52260(v2, v24, (__int64)&v69, v5, v22);
    v58[0] = v59;
    v69 = v71;
    v58[1] = 0x100000001LL;
    v25 = *(_QWORD *)a1;
    v66 = v68;
    v70 = 0x1000000000LL;
    v26 = *(_QWORD *)(v25 + 48);
    v27 = *(_QWORD *)(v25 + 40);
    v59[0] = v2;
    v67 = 0x1000000000LL;
    sub_11D0BA0(v58, v27, v26, v25, &v66, &v69);
    v28 = (__int64 *)v69;
    v29 = (__int64 *)&v69[8 * (unsigned int)v70];
    if ( v29 != (__int64 *)v69 )
    {
      do
      {
        v27 = *v28++;
        sub_F86EA0(a1, v27);
      }
      while ( v29 != v28 );
    }
    v30 = &v66[8 * (unsigned int)v67];
    if ( v30 != v66 )
    {
      v31 = v66;
      do
      {
        while ( 1 )
        {
          v32 = *(_QWORD **)v31;
          if ( !*(_QWORD *)(*(_QWORD *)v31 + 16LL) )
            break;
          v31 += 8;
          if ( v30 == v31 )
            goto LABEL_55;
        }
        v62 = *(_QWORD **)v31;
        v33 = v32;
        v60 = 0;
        v61 = 0;
        v34 = v32 + 512 != 0 && v32 + 1024 != 0;
        if ( v34 )
        {
          sub_BD73F0((__int64)&v60);
          v33 = v62;
        }
        v35 = *(_DWORD *)(a1 + 88);
        if ( v35 )
        {
          v36 = v35 - 1;
          v37 = *(_QWORD *)(a1 + 72);
          v27 = (v35 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
          v38 = (_QWORD *)(v37 + 24 * v27);
          v39 = (_QWORD *)v38[2];
          if ( v33 == v39 )
          {
LABEL_28:
            v63 = 0;
            v64 = 0;
            v65 = -8192;
            v40 = v38[2];
            if ( v40 != -8192 )
            {
              if ( v40 && v40 != -4096 )
              {
                v56 = v38;
                sub_BD60C0(v38);
                v38 = v56;
              }
              v38[2] = -8192;
              if ( v65 != 0 && v65 != -4096 && v65 != -8192 )
                sub_BD60C0(&v63);
            }
            --*(_DWORD *)(a1 + 80);
            v33 = v62;
            ++*(_DWORD *)(a1 + 84);
          }
          else
          {
            v52 = 1;
            while ( v39 != (_QWORD *)-4096LL )
            {
              v53 = v52 + 1;
              v27 = v36 & (unsigned int)(v52 + v27);
              v38 = (_QWORD *)(v37 + 24LL * (unsigned int)v27);
              v39 = (_QWORD *)v38[2];
              if ( v33 == v39 )
                goto LABEL_28;
              v52 = v53;
            }
          }
        }
        if ( v33 + 512 != 0 && v33 != 0 && v33 != (_QWORD *)-8192LL )
          sub_BD60C0(&v60);
        v60 = 0;
        v41 = v32;
        v61 = 0;
        v62 = v32;
        if ( v34 )
        {
          sub_BD73F0((__int64)&v60);
          v41 = v62;
        }
        v42 = *(_DWORD *)(a1 + 120);
        if ( v42 )
        {
          v27 = (unsigned int)(v42 - 1);
          v43 = *(_QWORD *)(a1 + 104);
          v44 = v27 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
          v45 = (_QWORD *)(v43 + 24LL * v44);
          v46 = (_QWORD *)v45[2];
          if ( v41 == v46 )
          {
LABEL_43:
            v63 = 0;
            v64 = 0;
            v65 = -8192;
            v47 = v45[2];
            if ( v47 != -8192 )
            {
              if ( v47 && v47 != -4096 )
                sub_BD60C0(v45);
              v45[2] = -8192;
              if ( v65 != -4096 && v65 != 0 && v65 != -8192 )
                sub_BD60C0(&v63);
            }
            --*(_DWORD *)(a1 + 112);
            v41 = v62;
            ++*(_DWORD *)(a1 + 116);
          }
          else
          {
            v51 = 1;
            while ( v46 != (_QWORD *)-4096LL )
            {
              v44 = v27 & (v51 + v44);
              v45 = (_QWORD *)(v43 + 24LL * v44);
              v46 = (_QWORD *)v45[2];
              if ( v46 == v41 )
                goto LABEL_43;
              ++v51;
            }
          }
        }
        LOBYTE(v27) = v41 != 0;
        if ( v41 + 512 != 0 && v41 != 0 && v41 != (_QWORD *)-8192LL )
          sub_BD60C0(&v60);
        v31 += 8;
        sub_B43D60(v32);
      }
      while ( v30 != v31 );
    }
LABEL_55:
    if ( (*(_BYTE *)(v57 + 7) & 0x40) != 0 )
      v48 = *(__int64 **)(v57 - 8);
    else
      v48 = (__int64 *)(v57 - 32LL * (*(_DWORD *)(v57 + 4) & 0x7FFFFFF));
    v2 = *v48;
    if ( v69 != v71 )
      _libc_free(v69, v27);
    if ( v66 != v68 )
      _libc_free(v66, v27);
    if ( (_QWORD *)v58[0] != v59 )
      _libc_free(v58[0], v27);
    sub_B43D60((_QWORD *)v57);
  }
  return v2;
}
