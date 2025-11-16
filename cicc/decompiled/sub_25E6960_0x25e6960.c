// Function: sub_25E6960
// Address: 0x25e6960
//
void __fastcall sub_25E6960(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v4; // r11
  __int64 *v5; // rbx
  __int64 v6; // r10
  __int64 v7; // rcx
  unsigned int v8; // esi
  __int64 v9; // rdx
  unsigned int v10; // r9d
  __int64 v11; // r8
  int v12; // ebx
  __int64 *v13; // r10
  unsigned int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // r13
  unsigned __int64 v17; // rbx
  int v18; // r13d
  __int64 *v19; // rdi
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r10
  unsigned __int64 v23; // rax
  __int64 *v24; // rbx
  __int64 v25; // r14
  unsigned int v26; // ecx
  __int64 v27; // rdi
  __int64 *v28; // rdx
  int v29; // r10d
  unsigned int v30; // r9d
  __int64 *v31; // rax
  __int64 v32; // r8
  unsigned __int64 v33; // r8
  unsigned int v34; // r13d
  unsigned int v35; // r10d
  _QWORD *v36; // rax
  __int64 v37; // r9
  unsigned int v38; // esi
  __int64 v39; // r12
  __int64 *v40; // r11
  int v41; // esi
  __int64 v42; // rax
  int v43; // esi
  __int64 v44; // r8
  unsigned int v45; // ecx
  __int64 v46; // rdi
  int v47; // esi
  int v48; // edi
  int v49; // r10d
  int v50; // r10d
  __int64 v51; // rdi
  unsigned int v52; // ecx
  int v53; // eax
  _QWORD *v54; // rdx
  __int64 v55; // rsi
  int v56; // eax
  int v57; // eax
  int v58; // ecx
  int v59; // ecx
  __int64 v60; // rdi
  _QWORD *v61; // r10
  unsigned int v62; // r13d
  int v63; // r9d
  __int64 v64; // rsi
  int v65; // edx
  int v66; // eax
  int v67; // eax
  int v68; // eax
  int v69; // edx
  int v70; // r13d
  __int64 *v71; // r10
  int v72; // edx
  int v73; // r13d
  _QWORD *v74; // r9
  __int64 *v75; // [rsp+0h] [rbp-90h]
  __int64 *v76; // [rsp+0h] [rbp-90h]
  unsigned __int64 v77; // [rsp+10h] [rbp-80h]
  int v78; // [rsp+10h] [rbp-80h]
  unsigned __int64 v79; // [rsp+10h] [rbp-80h]
  __int64 *v80; // [rsp+20h] [rbp-70h]
  __int64 v83; // [rsp+38h] [rbp-58h]
  unsigned int v84; // [rsp+38h] [rbp-58h]
  __int64 *v85; // [rsp+38h] [rbp-58h]
  __int64 *v86; // [rsp+38h] [rbp-58h]
  __int64 v87; // [rsp+48h] [rbp-48h] BYREF
  __int64 v88; // [rsp+50h] [rbp-40h] BYREF
  _QWORD v89[7]; // [rsp+58h] [rbp-38h] BYREF

  if ( a1 != a2 && a2 != a1 + 1 )
  {
    v4 = a1 + 1;
    while ( 1 )
    {
      v7 = *v4;
      v8 = *(_DWORD *)(a3 + 24);
      v9 = *a1;
      v87 = *v4;
      v88 = v9;
      if ( !v8 )
        break;
      v10 = v8 - 1;
      v11 = *(_QWORD *)(a3 + 8);
      v12 = 1;
      v13 = 0;
      v14 = (v8 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v15 = (__int64 *)(v11 + 16LL * v14);
      v16 = *v15;
      if ( v7 == *v15 )
      {
LABEL_9:
        v17 = v15[1];
        goto LABEL_10;
      }
      while ( v16 != -4096 )
      {
        if ( !v13 && v16 == -8192 )
          v13 = v15;
        v14 = v10 & (v12 + v14);
        v15 = (__int64 *)(v11 + 16LL * v14);
        v16 = *v15;
        if ( v7 == *v15 )
          goto LABEL_9;
        ++v12;
      }
      if ( !v13 )
        v13 = v15;
      v68 = *(_DWORD *)(a3 + 16);
      ++*(_QWORD *)a3;
      v69 = v68 + 1;
      v89[0] = v13;
      if ( 4 * (v68 + 1) >= 3 * v8 )
        goto LABEL_91;
      if ( v8 - *(_DWORD *)(a3 + 20) - v69 <= v8 >> 3 )
      {
        v86 = v4;
        goto LABEL_92;
      }
LABEL_83:
      *(_DWORD *)(a3 + 16) = v69;
      if ( *v13 != -4096 )
        --*(_DWORD *)(a3 + 20);
      *v13 = v7;
      v13[1] = 0;
      v8 = *(_DWORD *)(a3 + 24);
      if ( !v8 )
      {
        ++*(_QWORD *)a3;
        v17 = 0;
        v89[0] = 0;
        goto LABEL_87;
      }
      v11 = *(_QWORD *)(a3 + 8);
      v9 = v88;
      v10 = v8 - 1;
      v17 = 0;
LABEL_10:
      v18 = 1;
      v19 = 0;
      v20 = v10 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v21 = (__int64 *)(v11 + 16LL * v20);
      v22 = *v21;
      if ( *v21 != v9 )
      {
        while ( v22 != -4096 )
        {
          if ( !v19 && v22 == -8192 )
            v19 = v21;
          v20 = v10 & (v18 + v20);
          v21 = (__int64 *)(v11 + 16LL * v20);
          v22 = *v21;
          if ( *v21 == v9 )
            goto LABEL_11;
          ++v18;
        }
        if ( !v19 )
          v19 = v21;
        v66 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v67 = v66 + 1;
        v89[0] = v19;
        if ( 4 * v67 >= 3 * v8 )
        {
LABEL_87:
          v85 = v4;
          v8 *= 2;
        }
        else
        {
          if ( v8 - (v67 + *(_DWORD *)(a3 + 20)) > v8 >> 3 )
          {
LABEL_70:
            *(_DWORD *)(a3 + 16) = v67;
            if ( *v19 != -4096 )
              --*(_DWORD *)(a3 + 20);
            *v19 = v9;
            v23 = 0;
            v19[1] = 0;
            goto LABEL_12;
          }
          v85 = v4;
        }
        sub_9DDA50(a3, v8);
        sub_25E0C90(a3, &v88, v89);
        v9 = v88;
        v19 = (__int64 *)v89[0];
        v4 = v85;
        v67 = *(_DWORD *)(a3 + 16) + 1;
        goto LABEL_70;
      }
LABEL_11:
      v23 = v21[1];
LABEL_12:
      v6 = *v4;
      if ( v17 > v23 )
      {
        v5 = v4 + 1;
        if ( a1 != v4 )
        {
          v83 = *v4;
          memmove(a1 + 1, a1, (char *)v4 - (char *)a1);
          v6 = v83;
        }
        v4 = v5;
        *a1 = v6;
        if ( a2 == v5 )
          return;
      }
      else
      {
        v80 = v4;
        v24 = v4;
        v25 = *v4;
        v84 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
        while ( 1 )
        {
          v38 = *(_DWORD *)(a3 + 24);
          v39 = *(v24 - 1);
          v88 = v25;
          v40 = v24;
          if ( !v38 )
          {
            ++*(_QWORD *)a3;
            v89[0] = 0;
LABEL_21:
            sub_9DDA50(a3, 2 * v38);
            v41 = *(_DWORD *)(a3 + 24);
            v40 = v24;
            if ( v41 )
            {
              v42 = v88;
              v43 = v41 - 1;
              v44 = *(_QWORD *)(a3 + 8);
              v45 = v43 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
              v28 = (__int64 *)(v44 + 16LL * v45);
              v46 = *v28;
              if ( v88 == *v28 )
              {
LABEL_23:
                v47 = *(_DWORD *)(a3 + 16);
                v89[0] = v28;
                v48 = v47 + 1;
              }
              else
              {
                v70 = 1;
                v71 = 0;
                while ( v46 != -4096 )
                {
                  if ( v46 == -8192 && !v71 )
                    v71 = v28;
                  v45 = v43 & (v70 + v45);
                  v28 = (__int64 *)(v44 + 16LL * v45);
                  v46 = *v28;
                  if ( v88 == *v28 )
                    goto LABEL_23;
                  ++v70;
                }
                if ( !v71 )
                  v71 = v28;
                v72 = *(_DWORD *)(a3 + 16);
                v89[0] = v71;
                v48 = v72 + 1;
                v28 = v71;
              }
            }
            else
            {
              v65 = *(_DWORD *)(a3 + 16);
              v42 = v88;
              v89[0] = 0;
              v48 = v65 + 1;
              v28 = 0;
            }
            goto LABEL_24;
          }
          v26 = v38 - 1;
          v27 = *(_QWORD *)(a3 + 8);
          v28 = 0;
          v29 = 1;
          v30 = (v38 - 1) & v84;
          v31 = (__int64 *)(v27 + 16LL * v30);
          v32 = *v31;
          if ( *v31 == v25 )
          {
LABEL_15:
            v33 = v31[1];
            goto LABEL_16;
          }
          while ( v32 != -4096 )
          {
            if ( !v28 && v32 == -8192 )
              v28 = v31;
            v30 = v26 & (v29 + v30);
            v31 = (__int64 *)(v27 + 16LL * v30);
            v32 = *v31;
            if ( *v31 == v25 )
              goto LABEL_15;
            ++v29;
          }
          if ( !v28 )
            v28 = v31;
          v56 = *(_DWORD *)(a3 + 16);
          ++*(_QWORD *)a3;
          v48 = v56 + 1;
          v89[0] = v28;
          if ( 4 * (v56 + 1) >= 3 * v38 )
            goto LABEL_21;
          v42 = v25;
          if ( v38 - *(_DWORD *)(a3 + 20) - v48 <= v38 >> 3 )
          {
            sub_9DDA50(a3, v38);
            sub_25E0C90(a3, &v88, v89);
            v42 = v88;
            v40 = v24;
            v48 = *(_DWORD *)(a3 + 16) + 1;
            v28 = (__int64 *)v89[0];
          }
LABEL_24:
          *(_DWORD *)(a3 + 16) = v48;
          if ( *v28 != -4096 )
            --*(_DWORD *)(a3 + 20);
          *v28 = v42;
          v28[1] = 0;
          v38 = *(_DWORD *)(a3 + 24);
          if ( !v38 )
          {
            ++*(_QWORD *)a3;
            v33 = 0;
            goto LABEL_28;
          }
          v27 = *(_QWORD *)(a3 + 8);
          v26 = v38 - 1;
          v33 = 0;
LABEL_16:
          v34 = ((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4);
          v35 = v34 & v26;
          v36 = (_QWORD *)(v27 + 16LL * (v34 & v26));
          v37 = *v36;
          if ( *v36 != v39 )
            break;
LABEL_17:
          --v24;
          if ( v33 <= v36[1] )
            goto LABEL_33;
LABEL_18:
          v24[1] = *v24;
        }
        v78 = 1;
        v54 = 0;
        while ( v37 != -4096 )
        {
          if ( !v54 && v37 == -8192 )
            v54 = v36;
          v35 = v26 & (v78 + v35);
          v36 = (_QWORD *)(v27 + 16LL * v35);
          v37 = *v36;
          if ( v39 == *v36 )
            goto LABEL_17;
          ++v78;
        }
        if ( !v54 )
          v54 = v36;
        v57 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v53 = v57 + 1;
        if ( 4 * v53 >= 3 * v38 )
        {
LABEL_28:
          v75 = v40;
          v77 = v33;
          sub_9DDA50(a3, 2 * v38);
          v49 = *(_DWORD *)(a3 + 24);
          if ( !v49 )
            goto LABEL_126;
          v50 = v49 - 1;
          v51 = *(_QWORD *)(a3 + 8);
          v33 = v77;
          v40 = v75;
          v52 = v50 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
          v53 = *(_DWORD *)(a3 + 16) + 1;
          v54 = (_QWORD *)(v51 + 16LL * v52);
          v55 = *v54;
          if ( *v54 != v39 )
          {
            v73 = 1;
            v74 = 0;
            while ( v55 != -4096 )
            {
              if ( v74 || v55 != -8192 )
                v54 = v74;
              v52 = v50 & (v73 + v52);
              v55 = *(_QWORD *)(v51 + 16LL * v52);
              if ( v39 == v55 )
              {
                v54 = (_QWORD *)(v51 + 16LL * v52);
                goto LABEL_30;
              }
              ++v73;
              v74 = v54;
              v54 = (_QWORD *)(v51 + 16LL * v52);
            }
            if ( v74 )
              v54 = v74;
          }
        }
        else if ( v38 - (v53 + *(_DWORD *)(a3 + 20)) <= v38 >> 3 )
        {
          v76 = v40;
          v79 = v33;
          sub_9DDA50(a3, v38);
          v58 = *(_DWORD *)(a3 + 24);
          if ( !v58 )
          {
LABEL_126:
            ++*(_DWORD *)(a3 + 16);
            BUG();
          }
          v59 = v58 - 1;
          v60 = *(_QWORD *)(a3 + 8);
          v61 = 0;
          v62 = v59 & v34;
          v33 = v79;
          v40 = v76;
          v63 = 1;
          v53 = *(_DWORD *)(a3 + 16) + 1;
          v54 = (_QWORD *)(v60 + 16LL * v62);
          v64 = *v54;
          if ( *v54 != v39 )
          {
            while ( v64 != -4096 )
            {
              if ( !v61 && v64 == -8192 )
                v61 = v54;
              v62 = v59 & (v63 + v62);
              v54 = (_QWORD *)(v60 + 16LL * v62);
              v64 = *v54;
              if ( v39 == *v54 )
                goto LABEL_30;
              ++v63;
            }
            if ( v61 )
              v54 = v61;
          }
        }
LABEL_30:
        *(_DWORD *)(a3 + 16) = v53;
        if ( *v54 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v54 = v39;
        --v24;
        v54[1] = 0;
        if ( v33 )
          goto LABEL_18;
LABEL_33:
        *v40 = v25;
        v4 = v80 + 1;
        if ( a2 == v80 + 1 )
          return;
      }
    }
    ++*(_QWORD *)a3;
    v89[0] = 0;
LABEL_91:
    v86 = v4;
    v8 *= 2;
LABEL_92:
    sub_9DDA50(a3, v8);
    sub_25E0C90(a3, &v87, v89);
    v7 = v87;
    v13 = (__int64 *)v89[0];
    v4 = v86;
    v69 = *(_DWORD *)(a3 + 16) + 1;
    goto LABEL_83;
  }
}
