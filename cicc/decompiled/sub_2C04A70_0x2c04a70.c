// Function: sub_2C04A70
// Address: 0x2c04a70
//
void __fastcall sub_2C04A70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rdx
  __int64 v8; // rbx
  unsigned int v9; // r12d
  _QWORD *v10; // rax
  unsigned __int64 v11; // rcx
  _QWORD *v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  _QWORD *v15; // r14
  __int64 v16; // rax
  unsigned int v17; // r12d
  __int64 v18; // r13
  unsigned int *v19; // r10
  __int64 v20; // rbx
  unsigned int v21; // r13d
  _QWORD *v22; // r12
  unsigned int *v23; // r14
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  unsigned int v29; // eax
  _QWORD *v30; // rax
  __int64 v31; // r14
  __int64 v32; // r11
  unsigned int v33; // esi
  __int64 v34; // rbx
  unsigned int v35; // r10d
  unsigned int v36; // r9d
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 *v39; // rax
  __int64 v40; // rdx
  int v41; // esi
  int v42; // esi
  __int64 v43; // r8
  unsigned int v44; // ecx
  int v45; // eax
  __int64 *v46; // rdx
  __int64 v47; // rdi
  __int64 *v48; // r8
  int v49; // eax
  int v50; // ecx
  int v51; // ecx
  __int64 v52; // rdi
  __int64 *v53; // r9
  unsigned int v54; // r13d
  int v55; // r10d
  __int64 v56; // rsi
  int v57; // r13d
  __int64 *v58; // r10
  __int64 v59; // [rsp+8h] [rbp-1C8h]
  __int64 v60; // [rsp+18h] [rbp-1B8h]
  unsigned int v61; // [rsp+24h] [rbp-1ACh]
  __int64 v62; // [rsp+28h] [rbp-1A8h]
  int v63; // [rsp+28h] [rbp-1A8h]
  __int64 v64; // [rsp+28h] [rbp-1A8h]
  unsigned int *v65; // [rsp+38h] [rbp-198h]
  __int64 v66; // [rsp+38h] [rbp-198h]
  _QWORD *v67; // [rsp+40h] [rbp-190h] BYREF
  __int64 v68; // [rsp+48h] [rbp-188h]
  _QWORD v69[8]; // [rsp+50h] [rbp-180h] BYREF
  _BYTE *v70; // [rsp+90h] [rbp-140h] BYREF
  __int64 v71; // [rsp+98h] [rbp-138h]
  _BYTE v72[304]; // [rsp+A0h] [rbp-130h] BYREF

  v6 = a1;
  v7 = *(unsigned int *)(a1 + 8);
  v67 = v69;
  v61 = v7;
  v68 = 0x800000001LL;
  v69[0] = 0;
  if ( (unsigned int)v7 > 8 )
  {
    sub_C8D5F0((__int64)&v67, v69, v7, 8u, a5, a6);
    goto LABEL_3;
  }
  if ( (unsigned int)v7 > 1 )
  {
LABEL_3:
    v8 = 8;
    v9 = 1;
    do
    {
      v70 = *(_BYTE **)(*(_QWORD *)a1 + v8);
      v10 = sub_2C047F0(a1 + 528, (__int64 *)&v70);
      v11 = HIDWORD(v68);
      v12 = v10;
      v10[2] = *(_QWORD *)(*(_QWORD *)a1 + 8LL * *((unsigned int *)v10 + 1));
      v13 = (unsigned int)v68;
      v14 = (unsigned int)v68 + 1LL;
      if ( v14 > v11 )
      {
        sub_C8D5F0((__int64)&v67, v69, v14, 8u, a5, a6);
        v13 = (unsigned int)v68;
      }
      ++v9;
      v8 += 8;
      v67[v13] = v12;
      LODWORD(v68) = v68 + 1;
    }
    while ( v9 < v61 );
    v6 = a1;
    v15 = v67;
    v70 = v72;
    v71 = 0x2000000000LL;
    v16 = v61 - 1;
    if ( (unsigned int)v16 <= 1 )
      goto LABEL_30;
    goto LABEL_8;
  }
  v70 = v72;
  v71 = 0x2000000000LL;
  v16 = (unsigned int)(v7 - 1);
  if ( (unsigned int)v16 <= 1 )
    return;
  v15 = v69;
LABEL_8:
  v59 = v6;
  v60 = v16;
  v17 = v61;
  while ( 1 )
  {
    v18 = v15[v60];
    v19 = *(unsigned int **)(v18 + 24);
    *(_DWORD *)(v18 + 8) = *(_DWORD *)(v18 + 4);
    v65 = &v19[*(unsigned int *)(v18 + 32)];
    if ( v65 != v19 )
    {
      v20 = v18;
      v21 = v17;
      v22 = v15;
      v23 = v19;
      do
      {
        v24 = v22[*v23];
        v25 = (unsigned int)v71;
        if ( *(_DWORD *)(v24 + 4) < v21 )
        {
          v28 = *(unsigned int *)(v24 + 12);
        }
        else
        {
          do
          {
            if ( v25 + 1 > (unsigned __int64)HIDWORD(v71) )
            {
              sub_C8D5F0((__int64)&v70, v72, v25 + 1, 8u, a5, a6);
              v25 = (unsigned int)v71;
            }
            *(_QWORD *)&v70[8 * v25] = v24;
            v25 = (unsigned int)(v71 + 1);
            LODWORD(v71) = v71 + 1;
            v24 = v22[*(unsigned int *)(v24 + 4)];
          }
          while ( *(_DWORD *)(v24 + 4) >= v21 );
          v26 = v22[*(unsigned int *)(v24 + 12)];
          do
          {
            while ( 1 )
            {
              v27 = v24;
              v24 = *(_QWORD *)&v70[8 * (unsigned int)v25 - 8];
              LODWORD(v71) = v25 - 1;
              *(_DWORD *)(v24 + 4) = *(_DWORD *)(v27 + 4);
              if ( *(_DWORD *)(v26 + 8) >= *(_DWORD *)(v22[*(unsigned int *)(v24 + 12)] + 8LL) )
                break;
              *(_DWORD *)(v24 + 12) = *(_DWORD *)(v27 + 12);
              LODWORD(v25) = v71;
              if ( !(_DWORD)v71 )
                goto LABEL_19;
            }
            v26 = v22[*(unsigned int *)(v24 + 12)];
            LODWORD(v25) = v71;
          }
          while ( (_DWORD)v71 );
LABEL_19:
          v28 = *(unsigned int *)(v24 + 12);
          v22 = v67;
        }
        v29 = *(_DWORD *)(v22[v28] + 8LL);
        if ( *(_DWORD *)(v20 + 8) > v29 )
          *(_DWORD *)(v20 + 8) = v29;
        ++v23;
      }
      while ( v65 != v23 );
      v17 = v21;
    }
    --v17;
    --v60;
    if ( v17 == 2 )
      break;
    v15 = v67;
  }
  if ( v61 <= 2 )
    goto LABEL_27;
  v30 = v67;
  v66 = v59 + 528;
  v31 = 0;
  while ( 2 )
  {
    v32 = v30[v31 + 2];
    v33 = *(_DWORD *)(v59 + 552);
    v34 = *(_QWORD *)(v32 + 16);
    v35 = v33 - 1;
    v36 = *(_DWORD *)v30[*(unsigned int *)(v32 + 8)];
    while ( 1 )
    {
      if ( !v33 )
      {
        ++*(_QWORD *)(v59 + 528);
LABEL_41:
        v62 = v32;
        sub_2C04490(v66, 2 * v33);
        v41 = *(_DWORD *)(v59 + 552);
        if ( !v41 )
          goto LABEL_83;
        v42 = v41 - 1;
        v43 = *(_QWORD *)(v59 + 536);
        v32 = v62;
        v44 = v42 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v45 = *(_DWORD *)(v59 + 544) + 1;
        v46 = (__int64 *)(v43 + ((unsigned __int64)v44 << 6));
        v47 = *v46;
        if ( v34 != *v46 )
        {
          v57 = 1;
          v58 = 0;
          while ( v47 != -4096 )
          {
            if ( v47 == -8192 && !v58 )
              v58 = v46;
            v44 = v42 & (v57 + v44);
            v46 = (__int64 *)(v43 + ((unsigned __int64)v44 << 6));
            v47 = *v46;
            if ( v34 == *v46 )
              goto LABEL_43;
            ++v57;
          }
          if ( v58 )
            v46 = v58;
        }
        goto LABEL_43;
      }
      v37 = *(_QWORD *)(v59 + 536);
      v38 = v35 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v39 = (__int64 *)(v37 + ((unsigned __int64)v38 << 6));
      v40 = *v39;
      if ( v34 != *v39 )
        break;
LABEL_37:
      if ( v36 >= *((_DWORD *)v39 + 2) )
        goto LABEL_46;
      v34 = v39[3];
    }
    v63 = 1;
    v48 = 0;
    while ( v40 != -4096 )
    {
      if ( !v48 && v40 == -8192 )
        v48 = v39;
      v38 = v35 & (v63 + v38);
      v39 = (__int64 *)(v37 + ((unsigned __int64)v38 << 6));
      v40 = *v39;
      if ( v34 == *v39 )
        goto LABEL_37;
      ++v63;
    }
    v46 = v48;
    if ( !v48 )
      v46 = v39;
    v49 = *(_DWORD *)(v59 + 544);
    ++*(_QWORD *)(v59 + 528);
    v45 = v49 + 1;
    if ( 4 * v45 >= 3 * v33 )
      goto LABEL_41;
    if ( v33 - *(_DWORD *)(v59 + 548) - v45 <= v33 >> 3 )
    {
      v64 = v32;
      sub_2C04490(v66, v33);
      v50 = *(_DWORD *)(v59 + 552);
      if ( v50 )
      {
        v51 = v50 - 1;
        v52 = *(_QWORD *)(v59 + 536);
        v53 = 0;
        v54 = v51 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v32 = v64;
        v55 = 1;
        v45 = *(_DWORD *)(v59 + 544) + 1;
        v46 = (__int64 *)(v52 + ((unsigned __int64)v54 << 6));
        v56 = *v46;
        if ( v34 != *v46 )
        {
          while ( v56 != -4096 )
          {
            if ( v56 == -8192 && !v53 )
              v53 = v46;
            v54 = v51 & (v55 + v54);
            v46 = (__int64 *)(v52 + ((unsigned __int64)v54 << 6));
            v56 = *v46;
            if ( v34 == *v46 )
              goto LABEL_43;
            ++v55;
          }
          if ( v53 )
            v46 = v53;
        }
        goto LABEL_43;
      }
LABEL_83:
      ++*(_DWORD *)(v59 + 544);
      BUG();
    }
LABEL_43:
    *(_DWORD *)(v59 + 544) = v45;
    if ( *v46 != -4096 )
      --*(_DWORD *)(v59 + 548);
    *v46 = v34;
    *(_OWORD *)(v46 + 3) = 0;
    *(_OWORD *)(v46 + 5) = 0;
    v46[7] = 0;
    v46[4] = (__int64)(v46 + 6);
    v46[5] = 0x400000000LL;
    *(_OWORD *)(v46 + 1) = 0;
LABEL_46:
    ++v31;
    *(_QWORD *)(v32 + 16) = v34;
    if ( v61 > (int)v31 + 2 )
    {
      v30 = v67;
      continue;
    }
    break;
  }
LABEL_27:
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  v15 = v67;
LABEL_30:
  if ( v15 != v69 )
    _libc_free((unsigned __int64)v15);
}
