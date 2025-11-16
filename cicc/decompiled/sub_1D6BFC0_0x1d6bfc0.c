// Function: sub_1D6BFC0
// Address: 0x1d6bfc0
//
__int64 __fastcall sub_1D6BFC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r11
  __int64 v9; // r15
  __int64 v10; // r9
  __int64 v11; // r12
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 v15; // rbx
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  bool v19; // zf
  bool v20; // sf
  bool v21; // of
  unsigned int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // edi
  __int64 *v25; // rdx
  __int64 v26; // r11
  int v27; // r11d
  unsigned int v28; // edx
  __int64 *v29; // rax
  __int64 v30; // r10
  int v31; // eax
  __int64 v32; // r15
  __int64 *v33; // r12
  __int64 v34; // rax
  unsigned int v35; // esi
  unsigned int v36; // r8d
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 *v39; // rdx
  __int64 v40; // r10
  int v41; // ecx
  unsigned int v42; // r10d
  __int64 *v43; // rax
  __int64 v44; // r9
  int v45; // eax
  bool v46; // sf
  bool v47; // of
  __int64 v48; // rax
  __int64 *v49; // rax
  __int64 *v51; // r9
  int v52; // ecx
  __int64 *v53; // rdx
  __int64 v54; // rax
  int v55; // eax
  int v56; // eax
  unsigned int v57; // r8d
  __int64 *v58; // r10
  int v59; // edi
  int v60; // edi
  __int64 *v61; // rdi
  int v62; // eax
  int v63; // eax
  int v64; // eax
  __int64 *v65; // [rsp+8h] [rbp-98h]
  __int64 *v66; // [rsp+8h] [rbp-98h]
  int v67; // [rsp+10h] [rbp-90h]
  __int64 v68; // [rsp+10h] [rbp-90h]
  int v69; // [rsp+10h] [rbp-90h]
  __int64 v70; // [rsp+10h] [rbp-90h]
  __int64 v71; // [rsp+10h] [rbp-90h]
  __int64 v72; // [rsp+10h] [rbp-90h]
  __int64 v74; // [rsp+20h] [rbp-80h]
  __int64 v75; // [rsp+20h] [rbp-80h]
  __int64 v76; // [rsp+20h] [rbp-80h]
  __int64 *v77; // [rsp+20h] [rbp-80h]
  __int64 *v78; // [rsp+20h] [rbp-80h]
  int v80; // [rsp+28h] [rbp-78h]
  int v81; // [rsp+28h] [rbp-78h]
  int v82; // [rsp+28h] [rbp-78h]
  int v83; // [rsp+28h] [rbp-78h]
  __int64 v84; // [rsp+28h] [rbp-78h]
  __int64 v85; // [rsp+30h] [rbp-70h]
  __int64 v86; // [rsp+30h] [rbp-70h]
  unsigned int v87; // [rsp+38h] [rbp-68h]
  unsigned int v88; // [rsp+38h] [rbp-68h]
  int v89; // [rsp+38h] [rbp-68h]
  int v90; // [rsp+38h] [rbp-68h]
  __int64 v91; // [rsp+38h] [rbp-68h]
  __int64 v92; // [rsp+38h] [rbp-68h]
  __int64 v95; // [rsp+58h] [rbp-48h] BYREF
  __int64 v96; // [rsp+60h] [rbp-40h] BYREF
  _QWORD v97[7]; // [rsp+68h] [rbp-38h] BYREF

  v6 = a4;
  v85 = a3 & 1;
  if ( a2 < (a3 - 1) / 2 )
  {
    v9 = a2;
    v74 = a6 + 832;
    v10 = (a3 - 1) / 2;
    while ( 1 )
    {
      v11 = 2 * (v9 + 1);
      v15 = 32 * (v9 + 1);
      v16 = (__int64 *)(a1 + v15 - 16);
      v12 = (__int64 *)(a1 + v15);
      v17 = *v16;
      v18 = v16[1];
      v13 = *v12;
      if ( v17 == *v12 )
        goto LABEL_6;
      v21 = __OFSUB__(v18, v12[1]);
      v19 = v18 == v12[1];
      v20 = v18 - v12[1] < 0;
      if ( v18 == v12[1] )
        break;
LABEL_3:
      if ( !(v20 ^ v21 | v19) )
      {
        --v11;
        v12 = (__int64 *)(a1 + 16 * v11);
      }
      v13 = *v12;
LABEL_6:
      v14 = (__int64 *)(a1 + 16 * v9);
      *v14 = v13;
      v14[1] = v12[1];
      if ( v11 >= v10 )
      {
        v6 = a4;
        if ( v85 )
          goto LABEL_18;
        goto LABEL_34;
      }
      v9 = v11;
    }
    v22 = *(_DWORD *)(a6 + 856);
    v96 = *v12;
    if ( v22 )
    {
      v23 = *(_QWORD *)(a6 + 840);
      v87 = v22 - 1;
      v24 = (v22 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v25 = (__int64 *)(v23 + 16LL * v24);
      v26 = *v25;
      if ( v13 == *v25 )
      {
        v27 = *((_DWORD *)v25 + 2);
LABEL_13:
        v95 = v17;
        goto LABEL_14;
      }
      v67 = 1;
      v58 = 0;
      while ( v26 != -8 )
      {
        if ( v58 || v26 != -16 )
          v25 = v58;
        v24 = v87 & (v67 + v24);
        v66 = (__int64 *)(v23 + 16LL * v24);
        v26 = *v66;
        if ( v13 == *v66 )
        {
          v27 = *((_DWORD *)v66 + 2);
          goto LABEL_13;
        }
        ++v67;
        v58 = v25;
        v25 = (__int64 *)(v23 + 16LL * v24);
      }
      v59 = *(_DWORD *)(a6 + 848);
      if ( !v58 )
        v58 = v25;
      ++*(_QWORD *)(a6 + 832);
      v60 = v59 + 1;
      if ( 4 * v60 < 3 * v22 )
      {
        if ( v22 - *(_DWORD *)(a6 + 852) - v60 <= v22 >> 3 )
        {
          v72 = v10;
          v92 = v17;
          sub_1D6B640(v74, v22);
          sub_1D66AA0(v74, &v96, v97);
          v58 = (__int64 *)v97[0];
          v13 = v96;
          v10 = v72;
          v60 = *(_DWORD *)(a6 + 848) + 1;
          v17 = v92;
        }
        goto LABEL_67;
      }
    }
    else
    {
      ++*(_QWORD *)(a6 + 832);
    }
    v71 = v10;
    v91 = v17;
    sub_1D6B640(v74, 2 * v22);
    sub_1D66AA0(v74, &v96, v97);
    v58 = (__int64 *)v97[0];
    v13 = v96;
    v10 = v71;
    v60 = *(_DWORD *)(a6 + 848) + 1;
    v17 = v91;
LABEL_67:
    *(_DWORD *)(a6 + 848) = v60;
    if ( *v58 != -8 )
      --*(_DWORD *)(a6 + 852);
    *v58 = v13;
    *((_DWORD *)v58 + 2) = 0;
    v22 = *(_DWORD *)(a6 + 856);
    v95 = v17;
    v23 = *(_QWORD *)(a6 + 840);
    if ( !v22 )
    {
      ++*(_QWORD *)(a6 + 832);
      v27 = 0;
      goto LABEL_71;
    }
    v27 = 0;
    v87 = v22 - 1;
LABEL_14:
    v28 = v87 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v29 = (__int64 *)(v23 + 16LL * v28);
    v30 = *v29;
    if ( v17 == *v29 )
    {
      v31 = *((_DWORD *)v29 + 2);
LABEL_16:
      v21 = __OFSUB__(v31, v27);
      v19 = v31 == v27;
      v20 = v31 - v27 < 0;
      goto LABEL_3;
    }
    v69 = 1;
    v61 = 0;
    while ( v30 != -8 )
    {
      if ( v61 || v30 != -16 )
        v29 = v61;
      v28 = v87 & (v69 + v28);
      v65 = (__int64 *)(v23 + 16LL * v28);
      v30 = *v65;
      if ( v17 == *v65 )
      {
        v31 = *((_DWORD *)v65 + 2);
        goto LABEL_16;
      }
      ++v69;
      v61 = v29;
      v29 = (__int64 *)(v23 + 16LL * v28);
    }
    if ( !v61 )
      v61 = v29;
    v63 = *(_DWORD *)(a6 + 848);
    ++*(_QWORD *)(a6 + 832);
    v62 = v63 + 1;
    if ( 4 * v62 < 3 * v22 )
    {
      if ( v22 - (v62 + *(_DWORD *)(a6 + 852)) <= v22 >> 3 )
      {
        v70 = v10;
        v90 = v27;
        sub_1D6B640(v74, v22);
        sub_1D66AA0(v74, &v95, v97);
        v61 = (__int64 *)v97[0];
        v17 = v95;
        v10 = v70;
        v27 = v90;
        v62 = *(_DWORD *)(a6 + 848) + 1;
      }
      goto LABEL_72;
    }
LABEL_71:
    v68 = v10;
    v89 = v27;
    sub_1D6B640(v74, 2 * v22);
    sub_1D66AA0(v74, &v95, v97);
    v61 = (__int64 *)v97[0];
    v17 = v95;
    v27 = v89;
    v10 = v68;
    v62 = *(_DWORD *)(a6 + 848) + 1;
LABEL_72:
    *(_DWORD *)(a6 + 848) = v62;
    if ( *v61 != -8 )
      --*(_DWORD *)(a6 + 852);
    *v61 = v17;
    v31 = 0;
    *((_DWORD *)v61 + 2) = 0;
    goto LABEL_16;
  }
  v11 = a2;
  v12 = (__int64 *)(a1 + 16 * a2);
  if ( (a3 & 1) == 0 )
  {
LABEL_34:
    if ( (a3 - 2) / 2 == v11 )
    {
      v48 = v11 + 1;
      v11 = 2 * (v11 + 1) - 1;
      v49 = (__int64 *)(a1 + 32 * v48 - 16);
      *v12 = *v49;
      v12[1] = v49[1];
      v12 = (__int64 *)(a1 + 16 * v11);
    }
LABEL_18:
    v32 = (v11 - 1) / 2;
    if ( v11 > a2 )
    {
      v88 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
      v86 = a6 + 832;
      while ( 1 )
      {
        v12 = (__int64 *)(a1 + 16 * v32);
        v34 = *v12;
        if ( *v12 == v6 )
        {
          v12 = (__int64 *)(a1 + 16 * v11);
          break;
        }
        if ( v12[1] != a5 )
        {
          v33 = (__int64 *)(a1 + 16 * v11);
          if ( v12[1] >= a5 )
            goto LABEL_32;
          goto LABEL_21;
        }
        v35 = *(_DWORD *)(a6 + 856);
        v96 = *v12;
        if ( v35 )
        {
          v36 = v35 - 1;
          v37 = *(_QWORD *)(a6 + 840);
          v38 = (v35 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
          v39 = (__int64 *)(v37 + 16LL * v38);
          v40 = *v39;
          if ( v34 == *v39 )
          {
            v41 = *((_DWORD *)v39 + 2);
LABEL_28:
            v95 = v6;
            goto LABEL_29;
          }
          v80 = 1;
          v51 = 0;
          while ( v40 != -8 )
          {
            if ( v40 != -16 || v51 )
              v39 = v51;
            v38 = v36 & (v80 + v38);
            v77 = (__int64 *)(v37 + 16LL * v38);
            v40 = *v77;
            if ( v34 == *v77 )
            {
              v41 = *((_DWORD *)v77 + 2);
              goto LABEL_28;
            }
            ++v80;
            v51 = v39;
            v39 = (__int64 *)(v37 + 16LL * v38);
          }
          v52 = *(_DWORD *)(a6 + 848);
          if ( !v51 )
            v51 = v39;
          ++*(_QWORD *)(a6 + 832);
          if ( 4 * (v52 + 1) < 3 * v35 )
          {
            if ( v35 - *(_DWORD *)(a6 + 852) - (v52 + 1) > v35 >> 3 )
              goto LABEL_44;
            v84 = v6;
            goto LABEL_85;
          }
        }
        else
        {
          ++*(_QWORD *)(a6 + 832);
        }
        v84 = v6;
        v35 *= 2;
LABEL_85:
        sub_1D6B640(v86, v35);
        sub_1D66AA0(v86, &v96, v97);
        v51 = (__int64 *)v97[0];
        v34 = v96;
        v6 = v84;
LABEL_44:
        ++*(_DWORD *)(a6 + 848);
        if ( *v51 != -8 )
          --*(_DWORD *)(a6 + 852);
        *v51 = v34;
        *((_DWORD *)v51 + 2) = 0;
        v35 = *(_DWORD *)(a6 + 856);
        v95 = v6;
        v37 = *(_QWORD *)(a6 + 840);
        if ( !v35 )
        {
          ++*(_QWORD *)(a6 + 832);
          v41 = 0;
          goto LABEL_48;
        }
        v41 = 0;
        v36 = v35 - 1;
LABEL_29:
        v42 = v36 & v88;
        v43 = (__int64 *)(v37 + 16LL * (v36 & v88));
        v44 = *v43;
        if ( *v43 == v6 )
        {
          v45 = *((_DWORD *)v43 + 2);
          v47 = __OFSUB__(v41, v45);
          v46 = v41 - v45 < 0;
          goto LABEL_31;
        }
        v82 = 1;
        v53 = 0;
        while ( v44 != -8 )
        {
          if ( v44 != -16 || v53 )
            v43 = v53;
          v42 = v36 & (v82 + v42);
          v78 = (__int64 *)(v37 + 16LL * v42);
          v44 = *v78;
          if ( *v78 == v6 )
          {
            v64 = *((_DWORD *)v78 + 2);
            v47 = __OFSUB__(v41, v64);
            v46 = v41 - v64 < 0;
            goto LABEL_31;
          }
          ++v82;
          v53 = v43;
          v43 = (__int64 *)(v37 + 16LL * v42);
        }
        if ( !v53 )
          v53 = v43;
        v55 = *(_DWORD *)(a6 + 848);
        ++*(_QWORD *)(a6 + 832);
        v56 = v55 + 1;
        if ( 4 * v56 < 3 * v35 )
        {
          v57 = v35 - (*(_DWORD *)(a6 + 852) + v56);
          v54 = v6;
          if ( v57 <= v35 >> 3 )
          {
            v76 = v6;
            v83 = v41;
            sub_1D6B640(v86, v35);
            sub_1D66AA0(v86, &v95, v97);
            v53 = (__int64 *)v97[0];
            v54 = v95;
            v6 = v76;
            v41 = v83;
          }
          goto LABEL_49;
        }
LABEL_48:
        v75 = v6;
        v81 = v41;
        sub_1D6B640(v86, 2 * v35);
        sub_1D66AA0(v86, &v95, v97);
        v53 = (__int64 *)v97[0];
        v54 = v95;
        v41 = v81;
        v6 = v75;
LABEL_49:
        ++*(_DWORD *)(a6 + 848);
        if ( *v53 != -8 )
          --*(_DWORD *)(a6 + 852);
        *v53 = v54;
        *((_DWORD *)v53 + 2) = 0;
        v47 = 0;
        v46 = v41 < 0;
LABEL_31:
        v33 = (__int64 *)(a1 + 16 * v11);
        if ( v46 == v47 )
        {
LABEL_32:
          v12 = v33;
          break;
        }
LABEL_21:
        *v33 = *v12;
        v33[1] = v12[1];
        v11 = v32;
        if ( a2 >= v32 )
          break;
        v32 = (v32 - 1) / 2;
      }
    }
  }
  *v12 = v6;
  v12[1] = a5;
  return a5;
}
