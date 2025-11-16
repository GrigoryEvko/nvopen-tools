// Function: sub_353F520
// Address: 0x353f520
//
__int64 *__fastcall sub_353F520(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // r15
  unsigned int v13; // ebx
  __int64 v14; // r14
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // r15
  unsigned int v22; // r13d
  __int64 v23; // rbx
  unsigned int v24; // eax
  bool v25; // cf
  __int64 v26; // r9
  __int64 v27; // rsi
  __int64 v28; // r10
  _WORD *v29; // rdi
  __int64 v30; // rdx
  __int64 v31; // rcx
  unsigned __int16 *v32; // rdi
  unsigned __int16 *i; // rax
  _WORD *v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rdi
  unsigned __int16 *v37; // r8
  unsigned __int16 *v38; // rax
  __int64 v39; // rdi
  unsigned int v40; // r13d
  int v41; // edx
  __int64 v42; // rcx
  int v43; // edx
  unsigned int v44; // esi
  __int64 *v45; // rax
  __int64 v46; // rdi
  unsigned int v47; // esi
  unsigned int v48; // edi
  __int64 *v49; // rax
  __int64 v50; // r8
  int v52; // eax
  int v53; // r9d
  int v54; // eax
  int v55; // r8d
  __int64 v56; // [rsp+0h] [rbp-80h]
  unsigned int v57; // [rsp+0h] [rbp-80h]
  __int64 *v60; // [rsp+18h] [rbp-68h]
  __int64 v61; // [rsp+20h] [rbp-60h]
  __int64 v63; // [rsp+30h] [rbp-50h]
  __int64 v65; // [rsp+40h] [rbp-40h]
  __int64 v66; // [rsp+40h] [rbp-40h]
  __int64 v67; // [rsp+48h] [rbp-38h]

  v63 = a2;
  v67 = (a2 - 1) / 2;
  if ( a2 > a3 )
  {
    while ( 1 )
    {
      v60 = (__int64 *)(a1 + 8 * v67);
      v5 = *a5;
      v61 = *v60;
      v6 = *(unsigned __int16 *)(*(_QWORD *)(*v60 + 16) + 6LL);
      if ( *a5 )
      {
        v7 = *(_QWORD *)(v5 + 104);
        v65 = v7;
        if ( v7 )
          break;
      }
      v26 = a5[1];
      if ( !v26 || (v27 = *(_QWORD *)(v26 + 200), (v28 = *(_QWORD *)(v27 + 40)) == 0) )
        BUG();
      v29 = (_WORD *)(v28 + 14 * v6);
      if ( (*v29 & 0x1FFF) == 0x1FFF )
      {
        v11 = 0;
        v13 = -1;
      }
      else
      {
        v30 = (unsigned __int16)v29[1];
        v11 = 0;
        v13 = -1;
        v31 = *(_QWORD *)(v26 + 176);
        v32 = (unsigned __int16 *)(v31 + 6 * (v30 + (unsigned __int16)v29[2]));
        for ( i = (unsigned __int16 *)(v31 + 6 * v30); v32 != i; i += 3 )
        {
          if ( i[1] && *(_DWORD *)(*(_QWORD *)(v27 + 32) + 32LL * *i + 8) < v13 )
          {
            v11 = *i;
            v13 = *(_DWORD *)(*(_QWORD *)(v27 + 32) + 32 * v11 + 8);
          }
        }
      }
      v16 = *(unsigned __int16 *)(*(_QWORD *)(a4 + 16) + 6LL);
      if ( v5 )
      {
        v65 = *(_QWORD *)(v5 + 104);
        if ( v65 )
        {
          v56 = *(_QWORD *)(v5 + 80);
LABEL_11:
          v17 = v65 + 10 * v16;
          v18 = v56 + 24LL * *(unsigned __int16 *)(v17 + 4);
          v19 = v56 + 24LL * *(unsigned __int16 *)(v17 + 2);
          if ( v18 == v19 )
            goto LABEL_46;
          v20 = *(_QWORD *)(v19 + 8);
          v21 = v19 + 24;
          v66 = v20;
          v57 = v13;
          v22 = sub_39FAC40(v20);
          while ( v18 != v21 )
          {
            v23 = *(_QWORD *)(v21 + 8);
            v24 = sub_39FAC40(v23);
            if ( v24 < v22 )
            {
              v66 = v23;
              v22 = v24;
            }
            v21 += 24;
          }
          v25 = v22 < v57;
          if ( v22 != v57 )
            goto LABEL_18;
          goto LABEL_39;
        }
      }
      v34 = (_WORD *)(v28 + 14 * v16);
      if ( (*v34 & 0x1FFF) == 0x1FFF
        || (v35 = (unsigned __int16)v34[1],
            v36 = *(_QWORD *)(v26 + 176),
            v37 = (unsigned __int16 *)(v36 + 6 * (v35 + (unsigned __int16)v34[2])),
            v38 = (unsigned __int16 *)(v36 + 6 * v35),
            v37 == v38) )
      {
LABEL_46:
        if ( v13 != -1 )
          goto LABEL_44;
        v66 = 0;
        goto LABEL_39;
      }
      v39 = 0;
      v40 = -1;
      do
      {
        if ( v38[1] && *(_DWORD *)(*(_QWORD *)(v27 + 32) + 32LL * *v38 + 8) < v40 )
        {
          v39 = *v38;
          v40 = *(_DWORD *)(*(_QWORD *)(v27 + 32) + 32 * v39 + 8);
        }
        v38 += 3;
      }
      while ( v37 != v38 );
      v66 = v39;
      v25 = v40 < v13;
      if ( v40 != v13 )
      {
LABEL_18:
        if ( !v25 )
          goto LABEL_44;
        goto LABEL_19;
      }
LABEL_39:
      v41 = *((_DWORD *)a5 + 10);
      v42 = a5[3];
      if ( !v41 )
        goto LABEL_44;
      v43 = v41 - 1;
      v44 = v43 & (((0xBF58476D1CE4E5B9LL * v11) >> 31) ^ (484763065 * v11));
      v45 = (__int64 *)(v42 + 16LL * v44);
      v46 = *v45;
      if ( v11 == *v45 )
      {
LABEL_41:
        v47 = *((_DWORD *)v45 + 2);
      }
      else
      {
        v54 = 1;
        while ( v46 != -1 )
        {
          v55 = v54 + 1;
          v44 = v43 & (v54 + v44);
          v45 = (__int64 *)(v42 + 16LL * v44);
          v46 = *v45;
          if ( *v45 == v11 )
            goto LABEL_41;
          v54 = v55;
        }
        v47 = 0;
      }
      v48 = v43 & (((0xBF58476D1CE4E5B9LL * v66) >> 31) ^ (484763065 * v66));
      v49 = (__int64 *)(v42 + 16LL * v48);
      v50 = *v49;
      if ( v66 != *v49 )
      {
        v52 = 1;
        while ( v50 != -1 )
        {
          v53 = v52 + 1;
          v48 = v43 & (v52 + v48);
          v49 = (__int64 *)(v42 + 16LL * v48);
          v50 = *v49;
          if ( v66 == *v49 )
            goto LABEL_43;
          v52 = v53;
        }
        goto LABEL_44;
      }
LABEL_43:
      if ( *((_DWORD *)v49 + 2) <= v47 )
        goto LABEL_44;
LABEL_19:
      *(_QWORD *)(a1 + 8 * v63) = v61;
      v63 = v67;
      if ( a3 >= v67 )
        goto LABEL_45;
      v67 = (v67 - 1) / 2;
    }
    v8 = v7 + 10 * v6;
    v56 = *(_QWORD *)(v5 + 80);
    v9 = v56 + 24LL * *(unsigned __int16 *)(v8 + 4);
    v10 = v56 + 24LL * *(unsigned __int16 *)(v8 + 2);
    if ( v9 == v10 )
    {
      v11 = 0;
      v13 = -1;
    }
    else
    {
      v11 = *(_QWORD *)(v10 + 8);
      v12 = v10 + 24;
      v13 = sub_39FAC40(v11);
      while ( v9 != v12 )
      {
        v14 = *(_QWORD *)(v12 + 8);
        v15 = sub_39FAC40(v14);
        if ( v15 < v13 )
        {
          v11 = v14;
          v13 = v15;
        }
        v12 += 24;
      }
    }
    v16 = *(unsigned __int16 *)(*(_QWORD *)(a4 + 16) + 6LL);
    goto LABEL_11;
  }
LABEL_44:
  v60 = (__int64 *)(a1 + 8 * v63);
LABEL_45:
  *v60 = a4;
  return v60;
}
