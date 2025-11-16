// Function: sub_1E3D000
// Address: 0x1e3d000
//
__int64 __fastcall sub_1E3D000(unsigned int *a1, __int64 *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  char *v8; // rsi
  __int64 v9; // r12
  unsigned int v10; // eax
  unsigned int v11; // r13d
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  int v16; // r8d
  int v17; // r9d
  unsigned int v18; // r15d
  __int64 v19; // r13
  unsigned int v20; // r15d
  int v21; // eax
  int v22; // r10d
  __int64 *v23; // r9
  unsigned int k; // ecx
  __int64 v25; // rdi
  __int64 *v26; // r8
  __int64 v27; // rsi
  char v28; // al
  int v29; // eax
  unsigned int v30; // r13d
  unsigned int v31; // esi
  __int64 v32; // r8
  unsigned int v33; // edi
  unsigned int *v34; // rax
  unsigned int v35; // ecx
  int v36; // r11d
  unsigned int *v37; // rdx
  unsigned int v38; // eax
  int v39; // ecx
  _BYTE *v40; // rsi
  __int64 v41; // r15
  unsigned int v42; // r13d
  int v43; // eax
  int v44; // r11d
  unsigned int i; // r10d
  __int64 *v46; // rcx
  __int64 v47; // r9
  char v48; // al
  unsigned int v49; // eax
  __int64 result; // rax
  __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rcx
  int v54; // r8d
  int v55; // r9d
  unsigned int v56; // r15d
  __int64 v57; // r13
  unsigned int v58; // r15d
  int v59; // eax
  int v60; // r10d
  unsigned int j; // ecx
  __int64 v62; // rsi
  char v63; // al
  unsigned int v64; // eax
  unsigned int v65; // esi
  __int64 v66; // r8
  unsigned int v67; // eax
  unsigned int v68; // edi
  int v69; // r10d
  unsigned int *v70; // r9
  unsigned int v71; // ecx
  unsigned int v72; // eax
  unsigned int v73; // eax
  __int64 v74; // rdi
  int v75; // r9d
  unsigned int v76; // r15d
  unsigned int *v77; // r8
  unsigned int v78; // esi
  unsigned int v79; // r10d
  unsigned int v80; // ecx
  int v81; // [rsp+0h] [rbp-60h]
  __int64 *v82; // [rsp+0h] [rbp-60h]
  int v83; // [rsp+0h] [rbp-60h]
  __int64 *v84; // [rsp+8h] [rbp-58h]
  int v85; // [rsp+8h] [rbp-58h]
  unsigned int v86; // [rsp+8h] [rbp-58h]
  unsigned int v87; // [rsp+10h] [rbp-50h]
  __int64 *v88; // [rsp+10h] [rbp-50h]
  __int64 *v89; // [rsp+10h] [rbp-50h]
  __int64 *v90; // [rsp+18h] [rbp-48h]
  unsigned int v91; // [rsp+18h] [rbp-48h]
  __int64 *v92; // [rsp+18h] [rbp-48h]
  __int64 v93; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v94; // [rsp+28h] [rbp-38h]

  v8 = (char *)*((_QWORD *)a1 + 13);
  if ( v8 == *((char **)a1 + 14) )
  {
    sub_1E3B090((char **)a1 + 12, v8, a2);
  }
  else
  {
    if ( v8 )
    {
      *(_QWORD *)v8 = *a2;
      v8 = (char *)*((_QWORD *)a1 + 13);
    }
    v8 += 8;
    *((_QWORD *)a1 + 13) = v8;
  }
  v9 = *a2;
  v10 = a1[1];
  v11 = a1[8];
  v93 = v9;
  v94 = v10;
  if ( !v11 )
  {
    ++*((_QWORD *)a1 + 1);
    v12 = (__int64)(a1 + 2);
    goto LABEL_7;
  }
  v41 = *((_QWORD *)a1 + 2);
  v42 = v11 - 1;
  v43 = sub_1E1C690(&v93, (__int64)v8, a3, a4, a5, a6);
  v44 = 1;
  v26 = 0;
  for ( i = v42 & v43; ; i = v42 & v79 )
  {
    v46 = (__int64 *)(v41 + 16LL * i);
    v47 = *v46;
    if ( (unsigned __int64)(*v46 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v93 - 1) > 0xFFFFFFFFFFFFFFFDLL )
      break;
    v82 = v26;
    v85 = v44;
    v88 = (__int64 *)(v41 + 16LL * i);
    v91 = i;
    v48 = sub_1E15D60(v93, *v46, 3u);
    i = v91;
    v46 = v88;
    v44 = v85;
    v26 = v82;
    if ( v48 )
      goto LABEL_40;
    v47 = *v88;
LABEL_33:
    if ( !v47 )
    {
      v49 = a1[6];
      v11 = a1[8];
      v12 = (__int64)(a1 + 2);
      if ( !v26 )
        v26 = v46;
      ++*((_QWORD *)a1 + 1);
      v29 = v49 + 1;
      if ( 4 * v29 < 3 * v11 )
      {
        if ( v11 - (v29 + a1[7]) > v11 >> 3 )
        {
          v25 = v93;
          goto LABEL_14;
        }
        v51 = v11;
        sub_1E3CDC0(v12, v11);
        v56 = a1[8];
        if ( v56 )
        {
          v57 = *((_QWORD *)a1 + 2);
          v58 = v56 - 1;
          v59 = sub_1E1C690(&v93, v51, v52, v53, v54, v55);
          v23 = 0;
          v60 = 1;
          for ( j = v58 & v59; ; j = v58 & v80 )
          {
            v25 = v93;
            v26 = (__int64 *)(v57 + 16LL * j);
            v62 = *v26;
            if ( (unsigned __int64)(*v26 - 1) > 0xFFFFFFFFFFFFFFFDLL
              || (unsigned __int64)(v93 - 1) > 0xFFFFFFFFFFFFFFFDLL )
            {
              if ( v93 == v62 )
                goto LABEL_13;
            }
            else
            {
              v83 = v60;
              v86 = j;
              v89 = (__int64 *)(v57 + 16LL * j);
              v92 = v23;
              v63 = sub_1E15D60(v93, v62, 3u);
              v23 = v92;
              v26 = v89;
              j = v86;
              v60 = v83;
              if ( v63 )
              {
LABEL_12:
                v25 = v93;
LABEL_13:
                v29 = a1[6] + 1;
                goto LABEL_14;
              }
              v62 = *v89;
            }
            if ( !v62 )
              break;
            if ( v62 != -1 || v23 )
              v26 = v23;
            v80 = v60 + j;
            v23 = v26;
            ++v60;
          }
          v25 = v93;
          goto LABEL_56;
        }
LABEL_107:
        ++a1[6];
        BUG();
      }
LABEL_7:
      v13 = 2 * v11;
      sub_1E3CDC0(v12, v13);
      v18 = a1[8];
      if ( v18 )
      {
        v19 = *((_QWORD *)a1 + 2);
        v20 = v18 - 1;
        v21 = sub_1E1C690(&v93, v13, v14, v15, v16, v17);
        v22 = 1;
        v23 = 0;
        for ( k = v20 & v21; ; k = v20 & v71 )
        {
          v25 = v93;
          v26 = (__int64 *)(v19 + 16LL * k);
          v27 = *v26;
          if ( (unsigned __int64)(*v26 - 1) > 0xFFFFFFFFFFFFFFFDLL || (unsigned __int64)(v93 - 1) > 0xFFFFFFFFFFFFFFFDLL )
          {
            if ( v93 == v27 )
              goto LABEL_13;
          }
          else
          {
            v81 = v22;
            v84 = v23;
            v87 = k;
            v90 = (__int64 *)(v19 + 16LL * k);
            v28 = sub_1E15D60(v93, v27, 3u);
            v26 = v90;
            k = v87;
            v23 = v84;
            v22 = v81;
            if ( v28 )
              goto LABEL_12;
            v27 = *v90;
            v25 = v93;
          }
          if ( !v27 )
            break;
          if ( v23 || v27 != -1 )
            v26 = v23;
          v71 = v22 + k;
          v23 = v26;
          ++v22;
        }
LABEL_56:
        v29 = a1[6] + 1;
        if ( v23 )
          v26 = v23;
LABEL_14:
        a1[6] = v29;
        if ( *v26 )
          --a1[7];
        *v26 = v25;
        v30 = v94;
        *((_DWORD *)v26 + 2) = v94;
        v31 = a1[16];
        ++a1[1];
        LODWORD(v93) = v30;
        if ( v31 )
        {
          v32 = *((_QWORD *)a1 + 6);
          v33 = (v31 - 1) & (37 * v30);
          v34 = (unsigned int *)(v32 + 16LL * v33);
          v35 = *v34;
          if ( v30 == *v34 )
            goto LABEL_41;
          v36 = 1;
          v37 = 0;
          while ( v35 != -1 )
          {
            if ( v35 != -2 || v37 )
              v34 = v37;
            v33 = (v31 - 1) & (v36 + v33);
            v35 = *(_DWORD *)(v32 + 16LL * v33);
            if ( v30 == v35 )
              goto LABEL_41;
            ++v36;
            v37 = v34;
            v34 = (unsigned int *)(v32 + 16LL * v33);
          }
          if ( !v37 )
            v37 = v34;
          v38 = a1[14];
          ++*((_QWORD *)a1 + 5);
          v39 = v38 + 1;
          if ( 4 * (v38 + 1) < 3 * v31 )
          {
            if ( v31 - a1[15] - v39 > v31 >> 3 )
              goto LABEL_24;
            sub_1DF4FB0((__int64)(a1 + 10), v31);
            v72 = a1[16];
            if ( v72 )
            {
              v73 = v72 - 1;
              v74 = *((_QWORD *)a1 + 6);
              v75 = 1;
              v76 = v73 & (37 * v30);
              v77 = 0;
              v39 = a1[14] + 1;
              v37 = (unsigned int *)(v74 + 16LL * v76);
              v78 = *v37;
              if ( v30 != *v37 )
              {
                while ( v78 != -1 )
                {
                  if ( !v77 && v78 == -2 )
                    v77 = v37;
                  v76 = v73 & (v75 + v76);
                  v37 = (unsigned int *)(v74 + 16LL * v76);
                  v78 = *v37;
                  if ( v30 == *v37 )
                    goto LABEL_24;
                  ++v75;
                }
                if ( v77 )
                  v37 = v77;
              }
              goto LABEL_24;
            }
            goto LABEL_106;
          }
        }
        else
        {
          ++*((_QWORD *)a1 + 5);
        }
        sub_1DF4FB0((__int64)(a1 + 10), 2 * v31);
        v64 = a1[16];
        if ( v64 )
        {
          v65 = v64 - 1;
          v66 = *((_QWORD *)a1 + 6);
          v67 = (v64 - 1) & (37 * v30);
          v39 = a1[14] + 1;
          v37 = (unsigned int *)(v66 + 16LL * v67);
          v68 = *v37;
          if ( v30 != *v37 )
          {
            v69 = 1;
            v70 = 0;
            while ( v68 != -1 )
            {
              if ( v68 == -2 && !v70 )
                v70 = v37;
              v67 = v65 & (v69 + v67);
              v37 = (unsigned int *)(v66 + 16LL * v67);
              v68 = *v37;
              if ( v30 == *v37 )
                goto LABEL_24;
              ++v69;
            }
            if ( v70 )
              v37 = v70;
          }
LABEL_24:
          a1[14] = v39;
          if ( *v37 != -1 )
            --a1[15];
          *v37 = v30;
          *((_QWORD *)v37 + 1) = v9;
          v40 = (_BYTE *)*((_QWORD *)a1 + 10);
          if ( v40 != *((_BYTE **)a1 + 11) )
            goto LABEL_42;
          goto LABEL_27;
        }
LABEL_106:
        ++a1[14];
        BUG();
      }
      goto LABEL_107;
    }
    if ( !v26 && v47 == -1 )
      v26 = v46;
    v79 = v44 + i;
    ++v44;
  }
  if ( v93 != v47 )
    goto LABEL_33;
LABEL_40:
  LODWORD(v93) = *((_DWORD *)v46 + 2);
LABEL_41:
  v40 = (_BYTE *)*((_QWORD *)a1 + 10);
  if ( v40 == *((_BYTE **)a1 + 11) )
  {
LABEL_27:
    sub_B8BBF0((__int64)(a1 + 18), v40, &v93);
    goto LABEL_45;
  }
LABEL_42:
  if ( v40 )
  {
    *(_DWORD *)v40 = v93;
    v40 = (_BYTE *)*((_QWORD *)a1 + 10);
  }
  *((_QWORD *)a1 + 10) = v40 + 4;
LABEL_45:
  result = *a1;
  if ( a1[1] >= (unsigned int)result )
    sub_16BD130("Instruction mapping overflow!", 1u);
  return result;
}
