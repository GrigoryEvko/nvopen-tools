// Function: sub_1EC63D0
// Address: 0x1ec63d0
//
void __fastcall sub_1EC63D0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const void *a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int v10; // eax
  __int16 v11; // r13
  _WORD *v12; // rax
  _WORD *v13; // rdx
  unsigned __int16 v14; // r13
  __int64 v15; // rdx
  _WORD *v16; // r15
  __int16 v17; // ax
  __int64 v18; // rbx
  unsigned __int64 v19; // rbx
  _BYTE *v20; // rdi
  __int64 v21; // rbx
  __int64 v22; // r15
  __int64 v23; // r9
  unsigned int v24; // edi
  unsigned int *v25; // rax
  unsigned int v26; // ecx
  unsigned int v27; // esi
  __int64 v28; // r8
  unsigned int v29; // edi
  unsigned int *v30; // rax
  __int64 v31; // rdx
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // rax
  __int64 v35; // r14
  unsigned int v36; // r13d
  unsigned int v37; // esi
  unsigned int v38; // edx
  int v39; // r10d
  int v40; // r10d
  __int64 v41; // r8
  unsigned int v42; // r9d
  int v43; // edi
  unsigned int v44; // ecx
  int v45; // esi
  unsigned int *v46; // r11
  int v47; // r11d
  unsigned int *v48; // r10
  int v49; // edi
  int v50; // edi
  unsigned int *v51; // r11
  int v52; // ecx
  int v53; // r9d
  int v54; // r9d
  __int64 v55; // rcx
  int v56; // r11d
  unsigned int *v57; // r10
  int v58; // r10d
  int v59; // r10d
  __int64 v60; // r9
  unsigned int v61; // r8d
  int v62; // esi
  unsigned int v63; // ecx
  int v64; // r9d
  int v65; // r9d
  int v66; // r11d
  __int64 v67; // rsi
  int v69; // [rsp+8h] [rbp-B8h]
  int v70; // [rsp+10h] [rbp-B0h]
  unsigned int v73; // [rsp+20h] [rbp-A0h]
  int v74; // [rsp+20h] [rbp-A0h]
  unsigned int v75; // [rsp+20h] [rbp-A0h]
  const void *v76; // [rsp+30h] [rbp-90h]
  __int64 v77; // [rsp+30h] [rbp-90h]
  __int64 v78; // [rsp+38h] [rbp-88h]
  _BYTE *v79; // [rsp+40h] [rbp-80h] BYREF
  __int64 v80; // [rsp+48h] [rbp-78h]
  _BYTE v81[112]; // [rsp+50h] [rbp-70h] BYREF

  v7 = *(_QWORD *)(a1 + 920);
  v70 = *(_DWORD *)(v7 + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF) + 4);
  if ( !v70 )
  {
    v70 = *(_DWORD *)(a1 + 912);
    *(_DWORD *)(a1 + 912) = v70 + 1;
    *(_DWORD *)(v7 + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF) + 4) = v70;
  }
  v8 = *(_QWORD *)(a1 + 696);
  v79 = v81;
  v80 = 0x800000000LL;
  if ( !v8 )
    BUG();
  v9 = a3;
  v10 = *(_DWORD *)(*(_QWORD *)(v8 + 8) + 24LL * a3 + 16);
  v11 = a3 * (v10 & 0xF);
  v12 = (_WORD *)(*(_QWORD *)(v8 + 56) + 2LL * (v10 >> 4));
  v13 = v12 + 1;
  v14 = *v12 + v11;
LABEL_8:
  v16 = v13;
  while ( v16 )
  {
    v18 = sub_2103840(*(_QWORD *)(a1 + 272), a2, v14, v9, a5, a6);
    sub_20FD0B0(v18, 0xFFFFFFFFLL);
    v15 = (unsigned int)v80;
    a5 = *(const void **)(v18 + 112);
    v19 = *(unsigned int *)(v18 + 120);
    a6 = 8 * v19;
    if ( v19 > HIDWORD(v80) - (unsigned __int64)(unsigned int)v80 )
    {
      v76 = a5;
      sub_16CD150((__int64)&v79, v81, v19 + (unsigned int)v80, 8, (int)a5, a6);
      v15 = (unsigned int)v80;
      a5 = v76;
      a6 = 8 * v19;
    }
    if ( a6 )
    {
      memcpy(&v79[8 * v15], a5, a6);
      LODWORD(v15) = v80;
    }
    ++v16;
    LODWORD(v80) = v19 + v15;
    v17 = *(v16 - 1);
    v13 = 0;
    v14 += v17;
    if ( !v17 )
      goto LABEL_8;
  }
  v20 = v79;
  v21 = 0;
  if ( (_DWORD)v80 )
  {
    v22 = a4;
    v78 = 8LL * (unsigned int)v80;
    v77 = a1 + 952;
    while ( 1 )
    {
      v35 = *(_QWORD *)&v20[v21];
      v36 = *(_DWORD *)(v35 + 112);
      if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 264LL) + 4LL * (v36 & 0x7FFFFFFF)) )
        break;
LABEL_20:
      v21 += 8;
      if ( v78 == v21 )
        goto LABEL_31;
    }
    v37 = *(_DWORD *)(a1 + 976);
    v38 = *(_DWORD *)(a2 + 112);
    if ( v37 )
    {
      v23 = *(_QWORD *)(a1 + 960);
      v24 = (v37 - 1) & (37 * v36);
      v25 = (unsigned int *)(v23 + 12LL * v24);
      v26 = *v25;
      if ( v36 == *v25 )
      {
LABEL_15:
        v25[1] = v38;
        v27 = *(_DWORD *)(a1 + 976);
        if ( !v27 )
        {
LABEL_52:
          ++*(_QWORD *)(a1 + 952);
          goto LABEL_53;
        }
LABEL_16:
        v28 = *(_QWORD *)(a1 + 960);
        v29 = (v27 - 1) & (37 * v36);
        v30 = (unsigned int *)(v28 + 12LL * v29);
        v31 = *v30;
        if ( v36 == (_DWORD)v31 )
        {
LABEL_17:
          v30[2] = a3;
          sub_21031A0(*(_QWORD *)(a1 + 272), v35, v31, a3, v28);
          *(_DWORD *)(*(_QWORD *)(a1 + 920) + 8LL * (*(_DWORD *)(v35 + 112) & 0x7FFFFFFF) + 4) = v70;
          v34 = *(unsigned int *)(v22 + 8);
          if ( (unsigned int)v34 >= *(_DWORD *)(v22 + 12) )
          {
            sub_16CD150(v22, (const void *)(v22 + 16), 0, 4, v32, v33);
            v34 = *(unsigned int *)(v22 + 8);
          }
          *(_DWORD *)(*(_QWORD *)v22 + 4 * v34) = *(_DWORD *)(v35 + 112);
          v20 = v79;
          ++*(_DWORD *)(v22 + 8);
          goto LABEL_20;
        }
        v47 = 1;
        v48 = 0;
        while ( (_DWORD)v31 != -1 )
        {
          if ( !v48 && (_DWORD)v31 == -2 )
            v48 = v30;
          v29 = (v27 - 1) & (v47 + v29);
          v30 = (unsigned int *)(v28 + 12LL * v29);
          v31 = *v30;
          if ( v36 == (_DWORD)v31 )
            goto LABEL_17;
          ++v47;
        }
        v49 = *(_DWORD *)(a1 + 968);
        if ( v48 )
          v30 = v48;
        ++*(_QWORD *)(a1 + 952);
        v50 = v49 + 1;
        if ( 4 * v50 < 3 * v27 )
        {
          v31 = v27 - *(_DWORD *)(a1 + 972) - v50;
          v28 = v27 >> 3;
          if ( (unsigned int)v31 > (unsigned int)v28 )
            goto LABEL_40;
          sub_168FE70(v77, v27);
          v64 = *(_DWORD *)(a1 + 976);
          if ( !v64 )
          {
LABEL_96:
            ++*(_DWORD *)(a1 + 968);
            BUG();
          }
          v65 = v64 - 1;
          v66 = 1;
          v57 = 0;
          v67 = *(_QWORD *)(a1 + 960);
          v28 = v65 & (37 * v36);
          v30 = (unsigned int *)(v67 + 12 * v28);
          v50 = *(_DWORD *)(a1 + 968) + 1;
          v31 = *v30;
          if ( v36 == (_DWORD)v31 )
            goto LABEL_40;
          while ( (_DWORD)v31 != -1 )
          {
            if ( !v57 && (_DWORD)v31 == -2 )
              v57 = v30;
            v28 = v65 & (unsigned int)(v28 + v66);
            v30 = (unsigned int *)(v67 + 12 * v28);
            v31 = *v30;
            if ( v36 == (_DWORD)v31 )
              goto LABEL_40;
            ++v66;
          }
          goto LABEL_57;
        }
LABEL_53:
        sub_168FE70(v77, 2 * v27);
        v53 = *(_DWORD *)(a1 + 976);
        if ( !v53 )
          goto LABEL_96;
        v54 = v53 - 1;
        v55 = *(_QWORD *)(a1 + 960);
        v28 = v54 & (37 * v36);
        v50 = *(_DWORD *)(a1 + 968) + 1;
        v30 = (unsigned int *)(v55 + 12 * v28);
        v31 = *v30;
        if ( v36 == (_DWORD)v31 )
          goto LABEL_40;
        v56 = 1;
        v57 = 0;
        while ( (_DWORD)v31 != -1 )
        {
          if ( !v57 && (_DWORD)v31 == -2 )
            v57 = v30;
          v28 = v54 & (unsigned int)(v28 + v56);
          v30 = (unsigned int *)(v55 + 12 * v28);
          v31 = *v30;
          if ( v36 == (_DWORD)v31 )
            goto LABEL_40;
          ++v56;
        }
LABEL_57:
        if ( v57 )
          v30 = v57;
LABEL_40:
        *(_DWORD *)(a1 + 968) = v50;
        if ( *v30 != -1 )
          --*(_DWORD *)(a1 + 972);
        *v30 = v36;
        *(_QWORD *)(v30 + 1) = 0;
        goto LABEL_17;
      }
      v74 = 1;
      v51 = 0;
      while ( v26 != -1 )
      {
        if ( !v51 && v26 == -2 )
          v51 = v25;
        v24 = (v37 - 1) & (v74 + v24);
        v25 = (unsigned int *)(v23 + 12LL * v24);
        v26 = *v25;
        if ( v36 == *v25 )
          goto LABEL_15;
        ++v74;
      }
      v52 = *(_DWORD *)(a1 + 968);
      if ( v51 )
        v25 = v51;
      ++*(_QWORD *)(a1 + 952);
      v43 = v52 + 1;
      if ( 4 * (v52 + 1) < 3 * v37 )
      {
        if ( v37 - *(_DWORD *)(a1 + 972) - v43 <= v37 >> 3 )
        {
          v69 = 37 * v36;
          v75 = v38;
          sub_168FE70(v77, v37);
          v58 = *(_DWORD *)(a1 + 976);
          if ( !v58 )
          {
LABEL_97:
            ++*(_DWORD *)(a1 + 968);
            BUG();
          }
          v59 = v58 - 1;
          v46 = 0;
          v60 = *(_QWORD *)(a1 + 960);
          v38 = v75;
          v61 = v59 & v69;
          v43 = *(_DWORD *)(a1 + 968) + 1;
          v62 = 1;
          v25 = (unsigned int *)(v60 + 12LL * (v59 & (unsigned int)v69));
          v63 = *v25;
          if ( v36 != *v25 )
          {
            while ( v63 != -1 )
            {
              if ( v63 == -2 && !v46 )
                v46 = v25;
              v61 = v59 & (v62 + v61);
              v25 = (unsigned int *)(v60 + 12LL * v61);
              v63 = *v25;
              if ( v36 == *v25 )
                goto LABEL_49;
              ++v62;
            }
            goto LABEL_28;
          }
        }
LABEL_49:
        *(_DWORD *)(a1 + 968) = v43;
        if ( *v25 != -1 )
          --*(_DWORD *)(a1 + 972);
        *(_QWORD *)(v25 + 1) = 0;
        *v25 = v36;
        v25[1] = v38;
        v27 = *(_DWORD *)(a1 + 976);
        if ( !v27 )
          goto LABEL_52;
        goto LABEL_16;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 952);
    }
    v73 = v38;
    sub_168FE70(v77, 2 * v37);
    v39 = *(_DWORD *)(a1 + 976);
    if ( !v39 )
      goto LABEL_97;
    v40 = v39 - 1;
    v41 = *(_QWORD *)(a1 + 960);
    v38 = v73;
    v42 = v40 & (37 * v36);
    v43 = *(_DWORD *)(a1 + 968) + 1;
    v25 = (unsigned int *)(v41 + 12LL * v42);
    v44 = *v25;
    if ( v36 != *v25 )
    {
      v45 = 1;
      v46 = 0;
      while ( v44 != -1 )
      {
        if ( !v46 && v44 == -2 )
          v46 = v25;
        v42 = v40 & (v45 + v42);
        v25 = (unsigned int *)(v41 + 12LL * v42);
        v44 = *v25;
        if ( v36 == *v25 )
          goto LABEL_49;
        ++v45;
      }
LABEL_28:
      if ( v46 )
        v25 = v46;
      goto LABEL_49;
    }
    goto LABEL_49;
  }
LABEL_31:
  if ( v20 != v81 )
    _libc_free((unsigned __int64)v20);
}
