// Function: sub_FAF7E0
// Address: 0xfaf7e0
//
__int64 __fastcall sub_FAF7E0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 *v7; // r9
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rax
  _BYTE *v11; // r15
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // r14
  __int64 v15; // rax
  __int64 ***v16; // rbx
  __int64 **v17; // rax
  __int64 v18; // rcx
  _QWORD *v19; // rdx
  int v20; // esi
  unsigned int v21; // r8d
  __int64 *v22; // rdi
  __int64 v23; // r10
  __int64 **v24; // rdi
  __int64 v25; // r13
  __int64 *v26; // rbx
  __int64 v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // r12
  __int64 *v30; // rax
  int v31; // r11d
  __int64 **v32; // r10
  unsigned int v33; // edx
  __int64 **v34; // rdi
  __int64 *v35; // rcx
  char v36; // r13
  __int64 v37; // rdi
  __int64 *v38; // r14
  __int64 *v39; // rbx
  __int64 v40; // rdi
  int v41; // edx
  __int64 **v42; // rax
  unsigned int v43; // ecx
  __int64 *v44; // r8
  int v45; // edi
  __int64 **v46; // rsi
  __int64 v47; // rdx
  int v48; // r8d
  __int64 **v49; // rsi
  unsigned int v50; // ecx
  int v51; // ebx
  __int64 *v52; // r9
  int v53; // eax
  __int64 v54; // [rsp+18h] [rbp-C8h]
  __int64 v56; // [rsp+28h] [rbp-B8h]
  __int64 *v57; // [rsp+30h] [rbp-B0h] BYREF
  __int64 *v58; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v59; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v60; // [rsp+48h] [rbp-98h]
  __int64 v61; // [rsp+50h] [rbp-90h]
  __int64 v62; // [rsp+58h] [rbp-88h]
  __int64 v63; // [rsp+60h] [rbp-80h] BYREF
  __int64 v64; // [rsp+68h] [rbp-78h]
  __int64 **v65; // [rsp+70h] [rbp-70h]
  __int64 **v66; // [rsp+78h] [rbp-68h]
  _QWORD *v67; // [rsp+80h] [rbp-60h]
  unsigned __int64 v68; // [rsp+88h] [rbp-58h]
  __int64 **v69; // [rsp+90h] [rbp-50h]
  __int64 **v70; // [rsp+98h] [rbp-48h]
  _QWORD *v71; // [rsp+A0h] [rbp-40h]
  __int64 ***v72; // [rsp+A8h] [rbp-38h]

  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 1 )
    return 0;
  v3 = *(_BYTE **)(a1 - 96);
  if ( *v3 != 83 )
    return 0;
  v54 = *((_QWORD *)v3 - 8);
  if ( !v54 || **((_BYTE **)v3 - 4) != 18 )
    return 0;
  sub_B53900((__int64)v3);
  if ( !*(_BYTE *)(a2 + 192) )
    sub_CFDFC0(a2, a2, v4, v5, v6, v7);
  v8 = *(_QWORD *)(a2 + 16);
  v56 = v8 + 32LL * *(unsigned int *)(a2 + 24);
  if ( v56 == v8 )
    return 0;
  while ( 1 )
  {
    v9 = *(_QWORD *)(v8 + 16);
    if ( !v9 )
      goto LABEL_10;
    if ( *(_BYTE *)v9 != 85 )
      goto LABEL_10;
    v10 = *(_QWORD *)(v9 - 32);
    if ( !v10 )
      goto LABEL_10;
    if ( *(_BYTE *)v10 )
      goto LABEL_10;
    if ( *(_QWORD *)(v10 + 24) != *(_QWORD *)(v9 + 80) )
      goto LABEL_10;
    if ( (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
      goto LABEL_10;
    v11 = *(_BYTE **)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
    if ( *v11 != 83 )
      goto LABEL_10;
    v12 = *((_QWORD *)v11 - 8);
    if ( !v12 )
      goto LABEL_10;
    if ( **((_BYTE **)v11 - 4) != 18 )
      goto LABEL_10;
    sub_B53900(*(_QWORD *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF)));
    if ( v54 != v12 )
      goto LABEL_10;
    v13 = *(_QWORD *)(v8 + 16);
    v59 = 0;
    v60 = 0;
    v14 = *(__int64 **)(v13 + 40);
    v61 = 0;
    v62 = 0;
    v15 = *(_QWORD *)(a1 + 40);
    v63 = 0;
    v65 = 0;
    v57 = (__int64 *)v15;
    v66 = 0;
    v67 = 0;
    v68 = 0;
    v69 = 0;
    v70 = 0;
    v71 = 0;
    v72 = 0;
    v64 = 8;
    v63 = sub_22077B0(64);
    v16 = (__int64 ***)(v63 + ((4 * v64 - 4) & 0xFFFFFFFFFFFFFFF8LL));
    v17 = (__int64 **)sub_22077B0(512);
    v18 = (__int64)v57;
    v68 = (unsigned __int64)v16;
    v19 = v17 + 64;
    *v16 = v17;
    v66 = v17;
    v67 = v17 + 64;
    v72 = v16;
    v70 = v17;
    v71 = v17 + 64;
    v65 = v17;
    v69 = v17;
    if ( v14 == (__int64 *)v18 )
      goto LABEL_83;
    v20 = v62;
    if ( !(_DWORD)v62 )
    {
      ++v59;
      v58 = 0;
LABEL_95:
      v20 = 2 * v62;
LABEL_96:
      sub_E3B4A0((__int64)&v59, v20);
      sub_F9EAB0((__int64)&v59, (__int64 *)&v57, &v58);
      v18 = (__int64)v57;
      v22 = v58;
      v53 = v61 + 1;
      goto LABEL_91;
    }
    v21 = (v62 - 1) & (((unsigned int)v18 >> 4) ^ ((unsigned int)v18 >> 9));
    v22 = (__int64 *)(v60 + 8LL * v21);
    v23 = *v22;
    if ( v18 == *v22 )
      goto LABEL_24;
    v51 = 1;
    v52 = 0;
    while ( v23 != -4096 )
    {
      if ( v23 == -8192 && !v52 )
        v52 = v22;
      v21 = (v62 - 1) & (v51 + v21);
      v22 = (__int64 *)(v60 + 8LL * v21);
      v23 = *v22;
      if ( v18 == *v22 )
        goto LABEL_24;
      ++v51;
    }
    if ( v52 )
      v22 = v52;
    ++v59;
    v53 = v61 + 1;
    v58 = v22;
    if ( 4 * ((int)v61 + 1) >= (unsigned int)(3 * v62) )
      goto LABEL_95;
    if ( (int)v62 - HIDWORD(v61) - v53 <= (unsigned int)v62 >> 3 )
      goto LABEL_96;
LABEL_91:
    LODWORD(v61) = v53;
    if ( *v22 != -4096 )
      --HIDWORD(v61);
    *v22 = v18;
    v17 = v69;
    v19 = v71;
LABEL_24:
    if ( v17 == v19 - 1 )
    {
      v25 = v8;
      sub_F9EB60(&v63, &v57);
      v24 = v69;
    }
    else
    {
      if ( v17 )
      {
        *v17 = v57;
        v17 = v69;
      }
      v24 = v17 + 1;
      v25 = v8;
      v69 = v17 + 1;
    }
    while ( 2 )
    {
      if ( v65 == v24 )
      {
        v8 = v25;
LABEL_83:
        v36 = 1;
        goto LABEL_43;
      }
      if ( v70 == v24 )
      {
        v26 = (*(v72 - 1))[63];
        j_j___libc_free_0(v24, 512);
        v47 = (__int64)(*--v72 + 64);
        v70 = *v72;
        v71 = (_QWORD *)v47;
        v69 = v70 + 63;
      }
      else
      {
        v26 = *(v24 - 1);
        v69 = v24 - 1;
      }
      v27 = v26[2];
      if ( !v27 )
        break;
      while ( 1 )
      {
        v28 = *(_QWORD *)(v27 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v28 - 30) <= 0xAu )
          break;
        v27 = *(_QWORD *)(v27 + 8);
        if ( !v27 )
          goto LABEL_42;
      }
      v29 = 0;
LABEL_35:
      v30 = *(__int64 **)(v28 + 40);
      ++v29;
      v58 = v30;
      if ( v14 == v30 )
        goto LABEL_38;
      if ( !(_DWORD)v62 )
      {
        ++v59;
        goto LABEL_68;
      }
      v31 = 1;
      v32 = 0;
      v33 = (v62 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
      v34 = (__int64 **)(v60 + 8LL * v33);
      v35 = *v34;
      if ( v30 == *v34 )
        goto LABEL_38;
      while ( 1 )
      {
        if ( v35 == (__int64 *)-4096LL )
        {
          if ( !v32 )
            v32 = v34;
          ++v59;
          v41 = v61 + 1;
          if ( 4 * ((int)v61 + 1) >= (unsigned int)(3 * v62) )
          {
LABEL_68:
            sub_E3B4A0((__int64)&v59, 2 * v62);
            if ( (_DWORD)v62 )
            {
              v30 = v58;
              v43 = (v62 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
              v32 = (__int64 **)(v60 + 8LL * v43);
              v41 = v61 + 1;
              v44 = *v32;
              if ( *v32 != v58 )
              {
                v45 = 1;
                v46 = 0;
                while ( v44 != (__int64 *)-4096LL )
                {
                  if ( !v46 && v44 == (__int64 *)-8192LL )
                    v46 = v32;
                  v43 = (v62 - 1) & (v45 + v43);
                  v32 = (__int64 **)(v60 + 8LL * v43);
                  v44 = *v32;
                  if ( v58 == *v32 )
                    goto LABEL_59;
                  ++v45;
                }
                if ( v46 )
                  v32 = v46;
              }
              goto LABEL_59;
            }
          }
          else
          {
            if ( (int)v62 - HIDWORD(v61) - v41 > (unsigned int)v62 >> 3 )
            {
LABEL_59:
              LODWORD(v61) = v41;
              if ( *v32 != (__int64 *)-4096LL )
                --HIDWORD(v61);
              *v32 = v30;
              if ( (unsigned int)v61 <= 0x64 )
              {
                v42 = v69;
                if ( v69 == v71 - 1 )
                {
                  sub_F9EB60(&v63, &v58);
                }
                else
                {
                  if ( v69 )
                  {
                    *v69 = v58;
                    v42 = v69;
                  }
                  v69 = v42 + 1;
                }
                break;
              }
              goto LABEL_42;
            }
            sub_E3B4A0((__int64)&v59, v62);
            if ( (_DWORD)v62 )
            {
              v48 = 1;
              v41 = v61 + 1;
              v49 = 0;
              v50 = (v62 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
              v32 = (__int64 **)(v60 + 8LL * v50);
              v30 = *v32;
              if ( v58 != *v32 )
              {
                while ( v30 != (__int64 *)-4096LL )
                {
                  if ( v30 == (__int64 *)-8192LL && !v49 )
                    v49 = v32;
                  v50 = (v62 - 1) & (v48 + v50);
                  v32 = (__int64 **)(v60 + 8LL * v50);
                  v30 = *v32;
                  if ( v58 == *v32 )
                    goto LABEL_59;
                  ++v48;
                }
                v30 = v58;
                if ( v49 )
                  v32 = v49;
              }
              goto LABEL_59;
            }
          }
          LODWORD(v61) = v61 + 1;
          BUG();
        }
        if ( v35 != (__int64 *)-8192LL || v32 )
          v34 = v32;
        v33 = (v62 - 1) & (v31 + v33);
        v35 = *(__int64 **)(v60 + 8LL * v33);
        if ( v30 == v35 )
          break;
        ++v31;
        v32 = v34;
        v34 = (__int64 **)(v60 + 8LL * v33);
      }
LABEL_38:
      while ( 1 )
      {
        v27 = *(_QWORD *)(v27 + 8);
        if ( !v27 )
          break;
        v28 = *(_QWORD *)(v27 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v28 - 30) <= 0xAu )
          goto LABEL_35;
      }
      if ( v29 )
      {
        v24 = v69;
        continue;
      }
      break;
    }
LABEL_42:
    v8 = v25;
    v36 = 0;
LABEL_43:
    v37 = v63;
    if ( v63 )
    {
      v38 = (__int64 *)v68;
      v39 = (__int64 *)(v72 + 1);
      if ( (unsigned __int64)(v72 + 1) > v68 )
      {
        do
        {
          v40 = *v38++;
          j_j___libc_free_0(v40, 512);
        }
        while ( v39 > v38 );
        v37 = v63;
      }
      j_j___libc_free_0(v37, 8 * v64);
    }
    sub_C7D6A0(v60, 8LL * (unsigned int)v62, 8);
    if ( v36 )
      return 1;
LABEL_10:
    v8 += 32;
    if ( v56 == v8 )
      return 0;
  }
}
