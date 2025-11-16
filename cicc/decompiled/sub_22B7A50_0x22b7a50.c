// Function: sub_22B7A50
// Address: 0x22b7a50
//
__int64 __fastcall sub_22B7A50(int a1, int *a2, __int64 a3)
{
  unsigned int v4; // eax
  __int64 v5; // r9
  unsigned int v6; // esi
  int v7; // eax
  int *v8; // r11
  int v9; // edx
  __int64 v10; // r8
  unsigned int v11; // r12d
  int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rsi
  int *v15; // r12
  int v16; // eax
  unsigned int v17; // eax
  _DWORD *v18; // rax
  _DWORD *m; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned int v25; // r10d
  int *v26; // rsi
  int v27; // edi
  _DWORD *v28; // rdx
  _DWORD *v29; // rax
  unsigned int v30; // edx
  int *v31; // rdi
  int v32; // ecx
  int v33; // ebx
  int *v34; // rbx
  int v35; // eax
  unsigned int v36; // esi
  __int64 v37; // rdi
  unsigned int v38; // r9d
  unsigned int v39; // edx
  int *v40; // rcx
  int v41; // r8d
  __int64 v42; // rdx
  __int64 v43; // rsi
  int *v44; // r8
  int v45; // eax
  unsigned int v46; // edi
  int *v47; // rdx
  int v48; // ecx
  bool v49; // zf
  __int64 v50; // rax
  int v51; // edx
  __int64 v52; // r8
  int v53; // edx
  unsigned int v54; // edi
  int *v55; // rcx
  int v56; // esi
  int v57; // ecx
  unsigned int v58; // esi
  int v59; // edx
  int v60; // edx
  int v61; // r10d
  int k; // r11d
  int v63; // r11d
  int *v64; // r10
  int v65; // ecx
  int v66; // edx
  int i; // edx
  int v68; // r9d
  int j; // ecx
  int v70; // r9d
  int v71; // esi
  int v72; // r8d
  __int64 v73; // rsi
  int v74; // ecx
  unsigned int v76; // [rsp+28h] [rbp-D8h]
  __int64 v78; // [rsp+30h] [rbp-D0h]
  int v79; // [rsp+38h] [rbp-C8h]
  int v80; // [rsp+4Ch] [rbp-B4h] BYREF
  __int64 v81; // [rsp+50h] [rbp-B0h] BYREF
  _DWORD *v82; // [rsp+58h] [rbp-A8h]
  __int64 v83; // [rsp+60h] [rbp-A0h]
  unsigned int v84; // [rsp+68h] [rbp-98h]
  __int64 v85; // [rsp+70h] [rbp-90h] BYREF
  __int64 v86; // [rsp+78h] [rbp-88h] BYREF
  __int64 v87; // [rsp+80h] [rbp-80h]
  __int64 v88; // [rsp+88h] [rbp-78h]
  unsigned int v89; // [rsp+90h] [rbp-70h]
  _QWORD v90[4]; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int8 v91; // [rsp+C0h] [rbp-40h]

  v80 = *a2;
  v81 = 0;
  v4 = sub_AF1560(2u);
  v84 = v4;
  if ( !v4 )
  {
    v82 = 0;
    v5 = 1;
    v83 = 0;
LABEL_3:
    v90[0] = 0;
    v6 = 0;
    v81 = v5;
LABEL_4:
    sub_A08C50((__int64)&v81, 2 * v6);
    goto LABEL_5;
  }
  v83 = 0;
  v10 = sub_C7D670(4LL * v4, 4);
  v82 = (_DWORD *)v10;
  v28 = (_DWORD *)(v10 + 4LL * v84);
  v6 = v84;
  if ( (_DWORD *)v10 != v28 )
  {
    v29 = (_DWORD *)v10;
    do
    {
      if ( v29 )
        *v29 = -1;
      ++v29;
    }
    while ( v28 != v29 );
  }
  v5 = v81 + 1;
  if ( !v6 )
    goto LABEL_3;
  v7 = v80;
  v30 = (v6 - 1) & (37 * v80);
  v31 = (int *)(v10 + 4LL * v30);
  v32 = *v31;
  if ( v80 == *v31 )
    goto LABEL_9;
  v33 = 1;
  v8 = 0;
  while ( v32 != -1 )
  {
    if ( v8 || v32 != -2 )
      v31 = v8;
    v30 = (v6 - 1) & (v33 + v30);
    v32 = *(_DWORD *)(v10 + 4LL * v30);
    if ( v80 == v32 )
      goto LABEL_9;
    ++v33;
    v8 = v31;
    v31 = (int *)(v10 + 4LL * v30);
  }
  ++v81;
  if ( !v8 )
    v8 = v31;
  v9 = v83 + 1;
  v90[0] = v8;
  if ( 4 * ((int)v83 + 1) >= 3 * v6 )
    goto LABEL_4;
  if ( v6 - (v9 + HIDWORD(v83)) > v6 >> 3 )
    goto LABEL_6;
  sub_A08C50((__int64)&v81, v6);
LABEL_5:
  sub_22B31A0((__int64)&v81, &v80, v90);
  v7 = v80;
  v8 = (int *)v90[0];
  v9 = v83 + 1;
LABEL_6:
  LODWORD(v83) = v9;
  if ( *v8 != -1 )
    --HIDWORD(v83);
  *v8 = v7;
  v10 = (__int64)v82;
  v6 = v84;
  v5 = v81 + 1;
LABEL_9:
  v89 = v6;
  LODWORD(v85) = a1;
  v81 = v5;
  v87 = v10;
  v88 = v83;
  v86 = 1;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  sub_22B3900((__int64)v90, a3, (int *)&v85, (__int64)&v86);
  v11 = v91;
  v78 = v90[2];
  sub_C7D6A0(v87, 4LL * v89, 4);
  sub_C7D6A0((__int64)v82, 4LL * v84, 4);
  if ( !(_BYTE)v11 )
  {
    v23 = *(_QWORD *)(v78 + 16);
    v24 = *(unsigned int *)(v78 + 32);
    if ( !(_DWORD)v24 )
      return v11;
    v25 = (v24 - 1) & (37 * *a2);
    v26 = (int *)(v23 + 4LL * v25);
    v27 = *v26;
    if ( *v26 != *a2 )
    {
      v71 = 1;
      while ( v27 != -1 )
      {
        v72 = v71 + 1;
        v73 = ((_DWORD)v24 - 1) & (v25 + v71);
        v25 = v73;
        v26 = (int *)(v23 + 4 * v73);
        v27 = *v26;
        if ( *a2 == *v26 )
          goto LABEL_22;
        v71 = v72;
      }
      return v11;
    }
LABEL_22:
    if ( v26 == (int *)(v23 + 4 * v24) )
      return v11;
  }
  v11 = 1;
  v12 = *(_DWORD *)(v78 + 24);
  if ( v12 != 1 )
  {
    v13 = *(_QWORD *)(v78 + 16);
    v14 = 4LL * *(unsigned int *)(v78 + 32);
    v15 = (int *)(v13 + v14);
    if ( v12 && v15 != (int *)v13 )
    {
      v34 = *(int **)(v78 + 16);
      while ( (unsigned int)*v34 > 0xFFFFFFFD )
      {
        if ( ++v34 == v15 )
          goto LABEL_12;
      }
      if ( v15 != v34 )
      {
        v79 = 37 * a1;
        while ( 1 )
        {
          v35 = *v34;
          LODWORD(v81) = v35;
          if ( v35 == *a2 )
            goto LABEL_56;
          v36 = *(_DWORD *)(a3 + 24);
          v37 = *(_QWORD *)(a3 + 8);
          if ( !v36 )
            goto LABEL_56;
          v38 = v36 - 1;
          v39 = (v36 - 1) & (37 * v35);
          v40 = (int *)(v37 + 40LL * v39);
          v41 = *v40;
          if ( v35 == *v40 )
          {
LABEL_48:
            v42 = (unsigned int)v40[8];
            v43 = *((_QWORD *)v40 + 2);
            v44 = (int *)(v43 + 4 * v42);
            if ( !(_DWORD)v42 )
              goto LABEL_56;
            v45 = v42 - 1;
            v46 = (v42 - 1) & v79;
            v47 = (int *)(v43 + 4LL * v46);
            v48 = *v47;
            if ( *v47 != a1 )
            {
              for ( i = 1; ; i = v68 )
              {
                if ( v48 == -1 )
                  goto LABEL_56;
                v68 = i + 1;
                v46 = v45 & (i + v46);
                v47 = (int *)(v43 + 4LL * v46);
                v48 = *v47;
                if ( *v47 == a1 )
                  break;
              }
            }
            if ( v47 == v44 )
              goto LABEL_56;
            v49 = (unsigned __int8)sub_22B1BB0(a3, (int *)&v81, &v85) == 0;
            v50 = v85;
            if ( !v49 )
            {
              v51 = *(_DWORD *)(v85 + 32);
              v52 = *(_QWORD *)(v85 + 16);
              if ( !v51 )
                goto LABEL_56;
              v53 = v51 - 1;
              v54 = v53 & v79;
              v55 = (int *)(v52 + 4LL * (v53 & (unsigned int)v79));
              v56 = *v55;
              if ( *v55 != a1 )
              {
                for ( j = 1; ; j = v70 )
                {
                  if ( v56 == -1 )
                    goto LABEL_56;
                  v70 = j + 1;
                  v54 = v53 & (j + v54);
                  v55 = (int *)(v52 + 4LL * v54);
                  v56 = *v55;
                  if ( *v55 == a1 )
                    break;
                }
              }
              *v55 = -2;
              --*(_DWORD *)(v50 + 24);
              ++*(_DWORD *)(v50 + 28);
              goto LABEL_56;
            }
            v57 = *(_DWORD *)(a3 + 16);
            v58 = *(_DWORD *)(a3 + 24);
            v90[0] = v85;
            ++*(_QWORD *)a3;
            v59 = v57 + 1;
            if ( 4 * (v57 + 1) >= 3 * v58 )
            {
              v58 *= 2;
            }
            else if ( v58 - *(_DWORD *)(a3 + 20) - v59 > v58 >> 3 )
            {
LABEL_62:
              *(_DWORD *)(a3 + 16) = v59;
              if ( *(_DWORD *)v50 != -1 )
                --*(_DWORD *)(a3 + 20);
              v60 = v81;
              *(_QWORD *)(v50 + 8) = 0;
              *(_QWORD *)(v50 + 16) = 0;
              *(_DWORD *)v50 = v60;
              *(_QWORD *)(v50 + 24) = 0;
              *(_DWORD *)(v50 + 32) = 0;
              goto LABEL_56;
            }
            sub_22B36A0(a3, v58);
            sub_22B1BB0(a3, (int *)&v81, v90);
            v59 = *(_DWORD *)(a3 + 16) + 1;
            v50 = v90[0];
            goto LABEL_62;
          }
          v76 = (v36 - 1) & (37 * v35);
          v61 = *v40;
          for ( k = 1; ; ++k )
          {
            if ( v61 == -1 )
              goto LABEL_56;
            v76 = v38 & (v76 + k);
            v61 = *(_DWORD *)(v37 + 40LL * v76);
            if ( v35 == v61 )
              break;
          }
          v63 = 1;
          v64 = 0;
          while ( v41 != -1 )
          {
            if ( !v64 && v41 == -2 )
              v64 = v40;
            v74 = v63++;
            v39 = v38 & (v74 + v39);
            v40 = (int *)(v37 + 40LL * v39);
            v41 = *v40;
            if ( v35 == *v40 )
              goto LABEL_48;
          }
          if ( !v64 )
            v64 = v40;
          v65 = *(_DWORD *)(a3 + 16);
          ++*(_QWORD *)a3;
          v66 = v65 + 1;
          v90[0] = v64;
          if ( 4 * (v65 + 1) >= 3 * v36 )
          {
            v36 *= 2;
          }
          else if ( v36 - *(_DWORD *)(a3 + 20) - v66 > v36 >> 3 )
          {
            goto LABEL_74;
          }
          sub_22B36A0(a3, v36);
          sub_22B1BB0(a3, (int *)&v81, v90);
          v35 = v81;
          v64 = (int *)v90[0];
          v66 = *(_DWORD *)(a3 + 16) + 1;
LABEL_74:
          *(_DWORD *)(a3 + 16) = v66;
          if ( *v64 != -1 )
            --*(_DWORD *)(a3 + 20);
          *v64 = v35;
          *((_QWORD *)v64 + 1) = 0;
          *((_QWORD *)v64 + 2) = 0;
          *((_QWORD *)v64 + 3) = 0;
          v64[8] = 0;
          do
          {
LABEL_56:
            if ( ++v34 == v15 )
              goto LABEL_57;
          }
          while ( (unsigned int)*v34 > 0xFFFFFFFD );
          if ( v34 == v15 )
          {
LABEL_57:
            v13 = *(_QWORD *)(v78 + 16);
            v14 = 4LL * *(unsigned int *)(v78 + 32);
            break;
          }
        }
      }
    }
LABEL_12:
    sub_C7D6A0(v13, v14, 4);
    *(_DWORD *)v78 = -2;
    --*(_DWORD *)(a3 + 16);
    ++*(_DWORD *)(a3 + 20);
    v16 = *a2;
    v81 = 0;
    v80 = v16;
    v17 = sub_AF1560(2u);
    v84 = v17;
    if ( v17 )
    {
      v18 = (_DWORD *)sub_C7D670(4LL * v17, 4);
      v83 = 0;
      v82 = v18;
      for ( m = &v18[v84]; m != v18; ++v18 )
      {
        if ( v18 )
          *v18 = -1;
      }
    }
    else
    {
      v82 = 0;
      v83 = 0;
    }
    v11 = 1;
    sub_22B6470((__int64)v90, (__int64)&v81, &v80);
    v86 = 1;
    ++v81;
    LODWORD(v85) = a1;
    v20 = (__int64)v82;
    v82 = 0;
    v87 = v20;
    v21 = v83;
    v83 = 0;
    v88 = v21;
    LODWORD(v21) = v84;
    v84 = 0;
    v89 = v21;
    sub_22B3900((__int64)v90, a3, (int *)&v85, (__int64)&v86);
    sub_C7D6A0(v87, 4LL * v89, 4);
    sub_C7D6A0((__int64)v82, 4LL * v84, 4);
  }
  return v11;
}
