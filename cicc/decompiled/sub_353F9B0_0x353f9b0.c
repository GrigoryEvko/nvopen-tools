// Function: sub_353F9B0
// Address: 0x353f9b0
//
__int64 __fastcall sub_353F9B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r15
  unsigned int v14; // ebx
  __int64 v15; // r14
  unsigned int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // r15
  unsigned int v23; // r13d
  __int64 v24; // rbx
  unsigned int v25; // eax
  bool v26; // cf
  __int64 v27; // rax
  __int64 v28; // r9
  __int64 v29; // rsi
  __int64 v30; // r10
  _WORD *v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rcx
  unsigned __int16 *v34; // rdi
  unsigned __int16 *v35; // rax
  __int64 v36; // r11
  _WORD *v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rdi
  unsigned __int16 *v40; // r8
  unsigned __int16 *v41; // rax
  __int64 v42; // rdi
  unsigned int v43; // r13d
  int v44; // edx
  __int64 v45; // rcx
  int v46; // edx
  unsigned int v47; // esi
  __int64 *v48; // rax
  __int64 v49; // rdi
  unsigned int v50; // esi
  unsigned int v51; // edi
  __int64 *v52; // rax
  __int64 v53; // r8
  bool v54; // cc
  __int64 v55; // rax
  __int64 v56; // r15
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // rdx
  int v62; // eax
  int v63; // eax
  int v64; // r9d
  int v65; // r8d
  __int64 v69; // [rsp+20h] [rbp-B0h]
  unsigned int v70; // [rsp+20h] [rbp-B0h]
  __int64 v71; // [rsp+28h] [rbp-A8h]
  __int64 v74; // [rsp+48h] [rbp-88h]
  __int64 v75; // [rsp+48h] [rbp-88h]
  __int64 v76; // [rsp+50h] [rbp-80h]
  __int64 v77; // [rsp+58h] [rbp-78h]
  __int64 i; // [rsp+60h] [rbp-70h]
  __int64 v79; // [rsp+68h] [rbp-68h]
  __int64 v80[3]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v81; // [rsp+88h] [rbp-48h]
  __int64 v82; // [rsp+90h] [rbp-40h]
  unsigned int v83; // [rsp+98h] [rbp-38h]

  v71 = (a3 - 1) / 2;
  if ( a2 >= v71 )
  {
    v56 = a2;
    goto LABEL_57;
  }
  for ( i = a2; ; i = v5 )
  {
    v5 = 2 * i + 2;
    v77 = *(_QWORD *)(a1 + 8 * (2 * i + 1));
    v79 = *(_QWORD *)(a1 + 8 * v5);
    v6 = *(_QWORD *)a5;
    v7 = *(unsigned __int16 *)(*(_QWORD *)(v79 + 16) + 6LL);
    if ( *(_QWORD *)a5 )
    {
      v8 = *(_QWORD *)(v6 + 104);
      v74 = v8;
      if ( v8 )
      {
        v9 = v8 + 10 * v7;
        v69 = *(_QWORD *)(v6 + 80);
        v10 = v69 + 24LL * *(unsigned __int16 *)(v9 + 4);
        v11 = v69 + 24LL * *(unsigned __int16 *)(v9 + 2);
        if ( v10 == v11 )
        {
          v76 = 0;
          v14 = -1;
        }
        else
        {
          v12 = *(_QWORD *)(v11 + 8);
          v13 = v11 + 24;
          v76 = v12;
          v14 = sub_39FAC40(v12);
          while ( v13 != v10 )
          {
            v15 = *(_QWORD *)(v13 + 8);
            v16 = sub_39FAC40(v15);
            if ( v16 < v14 )
            {
              v76 = v15;
              v14 = v16;
            }
            v13 += 24;
          }
        }
        v17 = *(unsigned __int16 *)(*(_QWORD *)(v77 + 16) + 6LL);
        goto LABEL_12;
      }
    }
    v28 = *(_QWORD *)(a5 + 8);
    if ( !v28 || (v29 = *(_QWORD *)(v28 + 200), (v30 = *(_QWORD *)(v29 + 40)) == 0) )
      BUG();
    v31 = (_WORD *)(v30 + 14 * v7);
    if ( (*v31 & 0x1FFF) == 0x1FFF
      || (v32 = (unsigned __int16)v31[1],
          v33 = *(_QWORD *)(v28 + 176),
          v34 = (unsigned __int16 *)(v33 + 6 * (v32 + (unsigned __int16)v31[2])),
          v35 = (unsigned __int16 *)(v33 + 6 * v32),
          v34 == v35) )
    {
      v76 = 0;
      v14 = -1;
    }
    else
    {
      v36 = 0;
      v14 = -1;
      do
      {
        if ( v35[1] && *(_DWORD *)(*(_QWORD *)(v29 + 32) + 32LL * *v35 + 8) < v14 )
        {
          v36 = *v35;
          v14 = *(_DWORD *)(*(_QWORD *)(v29 + 32) + 32 * v36 + 8);
        }
        v35 += 3;
      }
      while ( v34 != v35 );
      v76 = v36;
    }
    v17 = *(unsigned __int16 *)(*(_QWORD *)(v77 + 16) + 6LL);
    if ( v6 )
    {
      v74 = *(_QWORD *)(v6 + 104);
      if ( v74 )
      {
        v69 = *(_QWORD *)(v6 + 80);
LABEL_12:
        v18 = v74 + 10 * v17;
        v19 = v69 + 24LL * *(unsigned __int16 *)(v18 + 2);
        v20 = v69 + 24LL * *(unsigned __int16 *)(v18 + 4);
        if ( v20 == v19 )
          goto LABEL_53;
        v21 = *(_QWORD *)(v19 + 8);
        v22 = v19 + 24;
        v75 = v21;
        v70 = v14;
        v23 = sub_39FAC40(v21);
        while ( v20 != v22 )
        {
          v24 = *(_QWORD *)(v22 + 8);
          v25 = sub_39FAC40(v24);
          if ( v25 < v23 )
          {
            v75 = v24;
            v23 = v25;
          }
          v22 += 24;
        }
        v26 = v23 < v70;
        if ( v23 != v70 )
          goto LABEL_19;
        goto LABEL_45;
      }
    }
    v37 = (_WORD *)(v30 + 14 * v17);
    if ( (*v37 & 0x1FFF) == 0x1FFF
      || (v38 = (unsigned __int16)v37[1],
          v39 = *(_QWORD *)(v28 + 176),
          v40 = (unsigned __int16 *)(v39 + 6 * (v38 + (unsigned __int16)v37[2])),
          v41 = (unsigned __int16 *)(v39 + 6 * v38),
          v40 == v41) )
    {
LABEL_53:
      if ( v14 != -1 )
        goto LABEL_23;
      v75 = 0;
      goto LABEL_45;
    }
    v42 = 0;
    v43 = -1;
    do
    {
      if ( v41[1] && *(_DWORD *)(*(_QWORD *)(v29 + 32) + 32LL * *v41 + 8) < v43 )
      {
        v42 = *v41;
        v43 = *(_DWORD *)(*(_QWORD *)(v29 + 32) + 32 * v42 + 8);
      }
      v41 += 3;
    }
    while ( v40 != v41 );
    v75 = v42;
    v26 = v43 < v14;
    if ( v43 != v14 )
    {
LABEL_19:
      v27 = v77;
      if ( v26 )
        v5 = 2 * i + 1;
      else
        v27 = v79;
      v79 = v27;
      goto LABEL_23;
    }
LABEL_45:
    v44 = *(_DWORD *)(a5 + 40);
    v45 = *(_QWORD *)(a5 + 24);
    if ( v44 )
    {
      v46 = v44 - 1;
      v47 = v46 & (((0xBF58476D1CE4E5B9LL * v76) >> 31) ^ (484763065 * v76));
      v48 = (__int64 *)(v45 + 16LL * v47);
      v49 = *v48;
      if ( *v48 == v76 )
      {
LABEL_47:
        v50 = *((_DWORD *)v48 + 2);
      }
      else
      {
        v62 = 1;
        while ( v49 != -1 )
        {
          v65 = v62 + 1;
          v47 = v46 & (v62 + v47);
          v48 = (__int64 *)(v45 + 16LL * v47);
          v49 = *v48;
          if ( *v48 == v76 )
            goto LABEL_47;
          v62 = v65;
        }
        v50 = 0;
      }
      v51 = v46 & (((0xBF58476D1CE4E5B9LL * v75) >> 31) ^ (484763065 * v75));
      v52 = (__int64 *)(v45 + 16LL * v51);
      v53 = *v52;
      if ( v75 == *v52 )
      {
LABEL_49:
        v54 = *((_DWORD *)v52 + 2) <= v50;
        v55 = v77;
        if ( v54 )
          v55 = v79;
        else
          v5 = 2 * i + 1;
        v79 = v55;
      }
      else
      {
        v63 = 1;
        while ( v53 != -1 )
        {
          v64 = v63 + 1;
          v51 = v46 & (v63 + v51);
          v52 = (__int64 *)(v45 + 16LL * v51);
          v53 = *v52;
          if ( *v52 == v75 )
            goto LABEL_49;
          v63 = v64;
        }
      }
    }
LABEL_23:
    *(_QWORD *)(a1 + 8 * i) = v79;
    if ( v5 >= v71 )
      break;
  }
  v56 = v5;
LABEL_57:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v56 )
  {
    *(_QWORD *)(a1 + 8 * v56) = *(_QWORD *)(a1 + 8 * (2 * v56 + 1));
    v56 = 2 * v56 + 1;
  }
  v80[2] = 1;
  v57 = *(_QWORD *)a5;
  v58 = *(_QWORD *)(a5 + 24);
  *(_QWORD *)(a5 + 24) = 0;
  ++*(_QWORD *)(a5 + 16);
  v80[0] = v57;
  v59 = *(_QWORD *)(a5 + 8);
  v81 = v58;
  v60 = *(_QWORD *)(a5 + 32);
  v80[1] = v59;
  LODWORD(v59) = *(_DWORD *)(a5 + 40);
  *(_QWORD *)(a5 + 32) = 0;
  *(_DWORD *)(a5 + 40) = 0;
  v82 = v60;
  v83 = v59;
  sub_353F520(a1, v56, a2, a4, v80);
  return sub_C7D6A0(v81, 16LL * v83, 8);
}
