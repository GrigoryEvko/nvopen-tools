// Function: sub_23CBE70
// Address: 0x23cbe70
//
__int64 __fastcall sub_23CBE70(__int64 a1, unsigned int a2, unsigned int a3)
{
  unsigned int v4; // r15d
  __int64 v5; // rax
  __int64 v6; // r13
  unsigned int v7; // esi
  int v8; // r12d
  __int64 v9; // rdx
  unsigned int v10; // r8d
  int v11; // r10d
  int v12; // r14d
  unsigned int v13; // edi
  unsigned int v14; // r11d
  int *v15; // rax
  int v16; // ecx
  int v17; // r9d
  __int64 v18; // r13
  unsigned int v19; // eax
  unsigned int v20; // edx
  int v21; // eax
  int v22; // eax
  int v24; // r14d
  int v25; // r8d
  int v26; // eax
  int v27; // eax
  unsigned __int64 v28; // r12
  unsigned int v29; // eax
  unsigned int v30; // esi
  _DWORD *v31; // r8
  __int64 v32; // r10
  unsigned int v33; // ecx
  _DWORD *v34; // rax
  int v35; // r9d
  __int64 *v36; // rax
  _DWORD *v37; // rdi
  int v38; // eax
  int v39; // eax
  int v40; // eax
  int v41; // edx
  int v42; // edx
  __int64 v43; // r10
  unsigned int v44; // esi
  int v45; // r9d
  int v46; // r14d
  _DWORD *v47; // r11
  int v48; // edx
  int v49; // edx
  __int64 v50; // r10
  int v51; // r14d
  unsigned int v52; // esi
  int v53; // r9d
  int v54; // r10d
  int *v55; // r9
  __int64 v56; // rdi
  int v57; // eax
  int v58; // eax
  int v59; // edx
  int v60; // edx
  __int64 v61; // rsi
  unsigned int v62; // r14d
  int v63; // ecx
  int v64; // r8d
  int *v65; // rdi
  int v66; // edx
  int v67; // edx
  __int64 v68; // rsi
  __int64 v69; // r14
  int v70; // r8d
  int v71; // ecx
  int v72; // [rsp+0h] [rbp-50h]
  int v73; // [rsp+0h] [rbp-50h]
  _DWORD *v74; // [rsp+0h] [rbp-50h]
  _DWORD *v75; // [rsp+0h] [rbp-50h]
  __int64 v76; // [rsp+8h] [rbp-48h]
  unsigned __int64 v77; // [rsp+10h] [rbp-40h]

  if ( !a3 )
    return 0;
  v77 = 0;
  v4 = a3;
  v76 = 4LL * a2;
  while ( 2 )
  {
    if ( *(_DWORD *)(a1 + 244) )
    {
LABEL_4:
      v5 = 4LL * *(unsigned int *)(a1 + 240);
      goto LABEL_5;
    }
    while ( 1 )
    {
      *(_DWORD *)(a1 + 240) = a2;
      v5 = v76;
LABEL_5:
      v6 = *(_QWORD *)(a1 + 232);
      v7 = *(_DWORD *)(v6 + 64);
      v8 = *(_DWORD *)(*(_QWORD *)a1 + v5);
      v9 = *(_QWORD *)(v6 + 48);
      if ( !v7 )
      {
LABEL_13:
        sub_23CBB90((_QWORD *)a1, v6, a2, v8);
        if ( v77 )
        {
          sub_23CC740(v77, *(_QWORD *)(a1 + 232));
          v77 = 0;
        }
        goto LABEL_15;
      }
      v10 = v7 - 1;
      v11 = 1;
      v12 = 37 * v8;
      v13 = (v7 - 1) & (37 * v8);
      v14 = v13;
      v15 = (int *)(v9 + 16LL * v13);
      v16 = *v15;
      v17 = *v15;
      if ( v8 == *v15 )
      {
LABEL_7:
        v18 = *((_QWORD *)v15 + 1);
        goto LABEL_8;
      }
      while ( 1 )
      {
        if ( v17 == -1 )
          goto LABEL_13;
        v14 = v10 & (v11 + v14);
        v17 = *(_DWORD *)(v9 + 16LL * v14);
        if ( v8 == v17 )
          break;
        ++v11;
      }
      v54 = 1;
      v55 = 0;
      while ( v16 != -1 )
      {
        if ( !v55 && v16 == -2 )
          v55 = v15;
        v13 = v10 & (v54 + v13);
        v15 = (int *)(v9 + 16LL * v13);
        v16 = *v15;
        if ( v8 == *v15 )
          goto LABEL_7;
        ++v54;
      }
      v56 = v6 + 40;
      if ( !v55 )
        v55 = v15;
      v57 = *(_DWORD *)(v6 + 56);
      ++*(_QWORD *)(v6 + 40);
      v58 = v57 + 1;
      if ( 4 * v58 >= 3 * v7 )
      {
        sub_23CAE70(v56, 2 * v7);
        v59 = *(_DWORD *)(v6 + 64);
        if ( !v59 )
          goto LABEL_104;
        v60 = v59 - 1;
        v61 = *(_QWORD *)(v6 + 48);
        v62 = v60 & v12;
        v58 = *(_DWORD *)(v6 + 56) + 1;
        v55 = (int *)(v61 + 16LL * v62);
        v63 = *v55;
        if ( v8 == *v55 )
          goto LABEL_66;
        v64 = 1;
        v65 = 0;
        while ( v63 != -1 )
        {
          if ( v63 == -2 && !v65 )
            v65 = v55;
          v62 = v60 & (v64 + v62);
          v55 = (int *)(v61 + 16LL * v62);
          v63 = *v55;
          if ( v8 == *v55 )
            goto LABEL_66;
          ++v64;
        }
      }
      else
      {
        if ( v7 - *(_DWORD *)(v6 + 60) - v58 > v7 >> 3 )
          goto LABEL_66;
        sub_23CAE70(v56, v7);
        v66 = *(_DWORD *)(v6 + 64);
        if ( !v66 )
        {
LABEL_104:
          ++*(_DWORD *)(v6 + 56);
          BUG();
        }
        v67 = v66 - 1;
        v68 = *(_QWORD *)(v6 + 48);
        v65 = 0;
        LODWORD(v69) = v67 & v12;
        v70 = 1;
        v58 = *(_DWORD *)(v6 + 56) + 1;
        v55 = (int *)(v68 + 16LL * (unsigned int)v69);
        v71 = *v55;
        if ( v8 == *v55 )
          goto LABEL_66;
        while ( v71 != -1 )
        {
          if ( v71 == -2 && !v65 )
            v65 = v55;
          v69 = v67 & (unsigned int)(v69 + v70);
          v55 = (int *)(v68 + 16 * v69);
          v71 = *v55;
          if ( v8 == *v55 )
            goto LABEL_66;
          ++v70;
        }
      }
      if ( v65 )
        v55 = v65;
LABEL_66:
      *(_DWORD *)(v6 + 56) = v58;
      if ( *v55 != -1 )
        --*(_DWORD *)(v6 + 60);
      *v55 = v8;
      v18 = 0;
      *((_QWORD *)v55 + 1) = 0;
LABEL_8:
      v19 = sub_23CA740(v18);
      v20 = *(_DWORD *)(a1 + 244);
      if ( v20 < v19 )
        break;
      *(_DWORD *)(a1 + 240) += v19;
      *(_DWORD *)(a1 + 244) = v20 - v19;
      v21 = *(_DWORD *)(a1 + 244);
      *(_QWORD *)(a1 + 232) = v18;
      if ( v21 )
        goto LABEL_4;
    }
    v24 = *(_DWORD *)(*(_QWORD *)a1 + v76);
    v25 = sub_23CC6E0(v18);
    v26 = *(_DWORD *)(a1 + 244);
    if ( *(_DWORD *)(*(_QWORD *)a1 + 4LL * (unsigned int)(v25 + v26)) != v24 )
    {
      v72 = *(_DWORD *)(a1 + 244) + sub_23CC6E0(v18) - 1;
      v27 = sub_23CC6E0(v18);
      v28 = sub_23CB860((_QWORD *)a1, *(_QWORD *)(a1 + 232), v27, v72, v8);
      sub_23CBB90((_QWORD *)a1, v28, a2, v24);
      sub_23CC6F0(v18, *(unsigned int *)(a1 + 244));
      v29 = sub_23CC6E0(v18);
      v30 = *(_DWORD *)(v28 + 64);
      v31 = (_DWORD *)(*(_QWORD *)a1 + 4LL * v29);
      if ( v30 )
      {
        v32 = *(_QWORD *)(v28 + 48);
        v33 = (v30 - 1) & (37 * *v31);
        v34 = (_DWORD *)(v32 + 16LL * v33);
        v35 = *v34;
        if ( *v31 == *v34 )
        {
LABEL_23:
          v36 = (__int64 *)(v34 + 2);
          goto LABEL_24;
        }
        v73 = 1;
        v37 = 0;
        while ( v35 != -1 )
        {
          if ( !v37 && v35 == -2 )
            v37 = v34;
          v33 = (v30 - 1) & (v73 + v33);
          v34 = (_DWORD *)(v32 + 16LL * v33);
          v35 = *v34;
          if ( *v31 == *v34 )
            goto LABEL_23;
          ++v73;
        }
        if ( !v37 )
          v37 = v34;
        v38 = *(_DWORD *)(v28 + 56);
        ++*(_QWORD *)(v28 + 40);
        v39 = v38 + 1;
        if ( 4 * v39 < 3 * v30 )
        {
          if ( v30 - *(_DWORD *)(v28 + 60) - v39 > v30 >> 3 )
          {
LABEL_35:
            *(_DWORD *)(v28 + 56) = v39;
            if ( *v37 != -1 )
              --*(_DWORD *)(v28 + 60);
            v40 = *v31;
            *((_QWORD *)v37 + 1) = 0;
            *v37 = v40;
            v36 = (__int64 *)(v37 + 2);
LABEL_24:
            *v36 = v18;
            if ( v77 )
              sub_23CC740(v77, v28);
            v77 = v28;
LABEL_15:
            --v4;
            if ( (unsigned __int8)sub_23CC720(*(_QWORD *)(a1 + 232)) )
            {
              v22 = *(_DWORD *)(a1 + 244);
              if ( v22 )
              {
                *(_DWORD *)(a1 + 244) = v22 - 1;
                *(_DWORD *)(a1 + 240) = a2 + 1 - v4;
              }
            }
            else
            {
              *(_QWORD *)(a1 + 232) = sub_23CC750(*(_QWORD *)(a1 + 232));
            }
            if ( !v4 )
              return 0;
            continue;
          }
          v75 = v31;
          sub_23CAE70(v28 + 40, v30);
          v48 = *(_DWORD *)(v28 + 64);
          if ( v48 )
          {
            v31 = v75;
            v49 = v48 - 1;
            v50 = *(_QWORD *)(v28 + 48);
            v47 = 0;
            v51 = 1;
            v52 = v49 & (37 * *v75);
            v39 = *(_DWORD *)(v28 + 56) + 1;
            v37 = (_DWORD *)(v50 + 16LL * v52);
            v53 = *v37;
            if ( *v75 == *v37 )
              goto LABEL_35;
            while ( v53 != -1 )
            {
              if ( !v47 && v53 == -2 )
                v47 = v37;
              v52 = v49 & (v51 + v52);
              v37 = (_DWORD *)(v50 + 16LL * v52);
              v53 = *v37;
              if ( *v75 == *v37 )
                goto LABEL_35;
              ++v51;
            }
            goto LABEL_48;
          }
          goto LABEL_105;
        }
      }
      else
      {
        ++*(_QWORD *)(v28 + 40);
      }
      v74 = v31;
      sub_23CAE70(v28 + 40, 2 * v30);
      v41 = *(_DWORD *)(v28 + 64);
      if ( v41 )
      {
        v31 = v74;
        v42 = v41 - 1;
        v43 = *(_QWORD *)(v28 + 48);
        v44 = v42 & (37 * *v74);
        v39 = *(_DWORD *)(v28 + 56) + 1;
        v37 = (_DWORD *)(v43 + 16LL * v44);
        v45 = *v37;
        if ( *v74 == *v37 )
          goto LABEL_35;
        v46 = 1;
        v47 = 0;
        while ( v45 != -1 )
        {
          if ( !v47 && v45 == -2 )
            v47 = v37;
          v44 = v42 & (v46 + v44);
          v37 = (_DWORD *)(v43 + 16LL * v44);
          v45 = *v37;
          if ( *v74 == *v37 )
            goto LABEL_35;
          ++v46;
        }
LABEL_48:
        if ( v47 )
          v37 = v47;
        goto LABEL_35;
      }
LABEL_105:
      ++*(_DWORD *)(v28 + 56);
      BUG();
    }
    break;
  }
  if ( v77 )
  {
    if ( !(unsigned __int8)sub_23CC720(*(_QWORD *)(a1 + 232)) )
      sub_23CC740(v77, *(_QWORD *)(a1 + 232));
    v26 = *(_DWORD *)(a1 + 244);
  }
  *(_DWORD *)(a1 + 244) = v26 + 1;
  return v4;
}
