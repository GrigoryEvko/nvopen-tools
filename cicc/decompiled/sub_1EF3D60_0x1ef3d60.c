// Function: sub_1EF3D60
// Address: 0x1ef3d60
//
__int64 __fastcall sub_1EF3D60(_QWORD **a1, __int64 a2, __m128i a3, __m128i a4)
{
  unsigned int v6; // esi
  __int64 *v7; // rdx
  unsigned int v8; // edi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 result; // rax
  __int64 v13; // r9
  int v14; // edx
  int v15; // esi
  _QWORD *v16; // r9
  unsigned int v17; // ecx
  int v18; // edx
  __int64 *v19; // r11
  __int64 v20; // rdi
  int i; // eax
  __int64 *v22; // rdi
  int v23; // r8d
  unsigned int v24; // r10d
  __int64 *v25; // rcx
  __int64 v26; // r14
  __int64 v27; // rsi
  __int64 v28; // rsi
  __int64 v29; // r12
  __int64 v30; // rdx
  __int64 v31; // rax
  _QWORD *v32; // r9
  __int64 v33; // rax
  _QWORD *v34; // r14
  __int64 v35; // r12
  int v36; // r8d
  int v37; // r9d
  __int64 v38; // r11
  __int64 v39; // rax
  __int64 *v40; // rdi
  _QWORD *v41; // r9
  __int64 v42; // rax
  _QWORD *v43; // r14
  __int64 v44; // r12
  int v45; // r8d
  int v46; // r9d
  __int64 v47; // r11
  __int64 v48; // rax
  __int64 v49; // rsi
  _QWORD *v50; // r9
  __int64 v51; // rax
  _QWORD *v52; // r14
  __int64 v53; // r12
  int v54; // r8d
  int v55; // r9d
  __int64 v56; // r11
  __int64 v57; // rax
  _QWORD *v58; // r9
  __int64 v59; // rax
  _QWORD *v60; // r14
  __int64 v61; // r12
  int v62; // r8d
  int v63; // r9d
  __int64 v64; // r11
  __int64 v65; // rax
  _QWORD *v66; // r9
  __int64 v67; // rax
  _QWORD *v68; // r14
  __int64 v69; // r12
  int v70; // r8d
  int v71; // r9d
  __int64 v72; // r11
  __int64 v73; // rax
  int v74; // edx
  int v75; // edx
  int v76; // ecx
  _QWORD *v77; // rdi
  __int64 *v78; // r9
  unsigned int v79; // r12d
  int v80; // r10d
  __int64 v81; // rsi
  int v82; // r10d
  int v83; // r12d
  __int64 *v84; // r10
  __int64 v85; // [rsp+0h] [rbp-70h]
  __int64 v86; // [rsp+0h] [rbp-70h]
  __int64 v87; // [rsp+0h] [rbp-70h]
  __int64 v88; // [rsp+0h] [rbp-70h]
  __int64 v89; // [rsp+0h] [rbp-70h]
  _QWORD *v90; // [rsp+10h] [rbp-60h]
  _QWORD *v91; // [rsp+10h] [rbp-60h]
  _QWORD *v92; // [rsp+10h] [rbp-60h]
  _QWORD *v93; // [rsp+10h] [rbp-60h]
  _QWORD *v94; // [rsp+10h] [rbp-60h]
  __int64 v95; // [rsp+18h] [rbp-58h]
  char v96; // [rsp+18h] [rbp-58h]
  char v97; // [rsp+18h] [rbp-58h]
  char v98; // [rsp+18h] [rbp-58h]
  char v99; // [rsp+18h] [rbp-58h]
  char v100; // [rsp+18h] [rbp-58h]
  __int64 v101; // [rsp+18h] [rbp-58h]
  __int64 v102; // [rsp+18h] [rbp-58h]
  __int64 *v103; // [rsp+20h] [rbp-50h] BYREF
  __int64 v104; // [rsp+28h] [rbp-48h]
  _BYTE v105[64]; // [rsp+30h] [rbp-40h] BYREF

  v6 = *((_DWORD *)a1 + 8);
  v7 = a1[2];
  if ( !v6 )
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0:
      case 0xB:
        v13 = (__int64)(a1 + 1);
        result = a2;
        goto LABEL_7;
      case 1:
LABEL_21:
        v27 = sub_1EF3D60(a1, *(_QWORD *)(a2 + 32), v7);
        if ( v27 == *(_QWORD *)(a2 + 32) )
          goto LABEL_84;
        result = sub_14835F0(*a1, v27, *(_QWORD *)(a2 + 40), 0, a3, a4);
        v7 = a1[2];
        v6 = *((_DWORD *)a1 + 8);
        break;
      case 2:
LABEL_23:
        v28 = sub_1EF3D60(a1, *(_QWORD *)(a2 + 32), v7);
        if ( v28 == *(_QWORD *)(a2 + 32) )
          goto LABEL_84;
        result = sub_14747F0((__int64)*a1, v28, *(_QWORD *)(a2 + 40), 0);
        v7 = a1[2];
        v6 = *((_DWORD *)a1 + 8);
        break;
      case 3:
LABEL_41:
        v49 = sub_1EF3D60(a1, *(_QWORD *)(a2 + 32), v7);
        if ( v49 == *(_QWORD *)(a2 + 32) )
          goto LABEL_84;
        result = sub_147B0D0((__int64)*a1, v49, *(_QWORD *)(a2 + 40), 0);
        v7 = a1[2];
        v6 = *((_DWORD *)a1 + 8);
        break;
      case 4:
LABEL_34:
        v41 = *(_QWORD **)(a2 + 32);
        v104 = 0x200000000LL;
        v42 = *(_QWORD *)(a2 + 40);
        v103 = (__int64 *)v105;
        v91 = &v41[v42];
        if ( v41 == v91 )
          goto LABEL_85;
        v97 = 0;
        v43 = v41;
        do
        {
          v44 = *v43;
          v47 = sub_1EF3D60(a1, *v43, v7);
          v48 = (unsigned int)v104;
          if ( (unsigned int)v104 >= HIDWORD(v104) )
          {
            v88 = v47;
            sub_16CD150((__int64)&v103, v105, 0, 8, v45, v46);
            v48 = (unsigned int)v104;
            v47 = v88;
          }
          v7 = v103;
          v103[v48] = v47;
          v40 = v103;
          LODWORD(v104) = v104 + 1;
          ++v43;
          v97 |= v103[(unsigned int)v104 - 1] != v44;
        }
        while ( v91 != v43 );
        result = a2;
        if ( v97 )
        {
          result = (__int64)sub_147DD40((__int64)*a1, (__int64 *)&v103, 0, 0, a3, a4);
          v40 = v103;
        }
        goto LABEL_64;
      case 5:
LABEL_57:
        v66 = *(_QWORD **)(a2 + 32);
        v104 = 0x200000000LL;
        v67 = *(_QWORD *)(a2 + 40);
        v103 = (__int64 *)v105;
        v94 = &v66[v67];
        if ( v66 == v94 )
          goto LABEL_85;
        v100 = 0;
        v68 = v66;
        do
        {
          v69 = *v68;
          v72 = sub_1EF3D60(a1, *v68, v7);
          v73 = (unsigned int)v104;
          if ( (unsigned int)v104 >= HIDWORD(v104) )
          {
            v87 = v72;
            sub_16CD150((__int64)&v103, v105, 0, 8, v70, v71);
            v73 = (unsigned int)v104;
            v72 = v87;
          }
          v7 = v103;
          v103[v73] = v72;
          v40 = v103;
          LODWORD(v104) = v104 + 1;
          ++v68;
          v100 |= v103[(unsigned int)v104 - 1] != v69;
        }
        while ( v94 != v68 );
        result = a2;
        if ( v100 )
        {
          result = sub_147EE30(*a1, &v103, 0, 0, a3, a4);
          v40 = v103;
        }
        goto LABEL_64;
      case 6:
LABEL_25:
        v29 = sub_1EF3D60(a1, *(_QWORD *)(a2 + 32), v7);
        v31 = sub_1EF3D60(a1, *(_QWORD *)(a2 + 40), v30);
        if ( v29 == *(_QWORD *)(a2 + 32) && v31 == *(_QWORD *)(a2 + 40) )
        {
LABEL_84:
          v7 = a1[2];
          v6 = *((_DWORD *)a1 + 8);
LABEL_85:
          result = a2;
        }
        else
        {
          result = sub_1483CF0(*a1, v29, v31, a3, a4);
          v7 = a1[2];
          v6 = *((_DWORD *)a1 + 8);
        }
        break;
      case 7:
LABEL_27:
        v32 = *(_QWORD **)(a2 + 32);
        v104 = 0x200000000LL;
        v33 = *(_QWORD *)(a2 + 40);
        v103 = (__int64 *)v105;
        v90 = &v32[v33];
        if ( v32 == v90 )
          goto LABEL_85;
        v96 = 0;
        v34 = v32;
        do
        {
          v35 = *v34;
          v38 = sub_1EF3D60(a1, *v34, v7);
          v39 = (unsigned int)v104;
          if ( (unsigned int)v104 >= HIDWORD(v104) )
          {
            v89 = v38;
            sub_16CD150((__int64)&v103, v105, 0, 8, v36, v37);
            v39 = (unsigned int)v104;
            v38 = v89;
          }
          v7 = v103;
          v103[v39] = v38;
          v40 = v103;
          LODWORD(v104) = v104 + 1;
          ++v34;
          v96 |= v103[(unsigned int)v104 - 1] != v35;
        }
        while ( v90 != v34 );
        result = a2;
        if ( v96 )
        {
          result = sub_14785F0((__int64)*a1, &v103, *(_QWORD *)(a2 + 48), *(_WORD *)(a2 + 26) & 7);
          v40 = v103;
        }
        goto LABEL_64;
      case 8:
LABEL_43:
        v50 = *(_QWORD **)(a2 + 32);
        v104 = 0x200000000LL;
        v51 = *(_QWORD *)(a2 + 40);
        v103 = (__int64 *)v105;
        v92 = &v50[v51];
        if ( v50 == v92 )
          goto LABEL_85;
        v98 = 0;
        v52 = v50;
        do
        {
          v53 = *v52;
          v56 = sub_1EF3D60(a1, *v52, v7);
          v57 = (unsigned int)v104;
          if ( (unsigned int)v104 >= HIDWORD(v104) )
          {
            v85 = v56;
            sub_16CD150((__int64)&v103, v105, 0, 8, v54, v55);
            v57 = (unsigned int)v104;
            v56 = v85;
          }
          v7 = v103;
          v103[v57] = v56;
          v40 = v103;
          LODWORD(v104) = v104 + 1;
          ++v52;
          v98 |= v103[(unsigned int)v104 - 1] != v53;
        }
        while ( v92 != v52 );
        result = a2;
        if ( v98 )
        {
          result = sub_14813B0(*a1, &v103, a3, a4);
          v40 = v103;
        }
        goto LABEL_64;
      case 9:
LABEL_50:
        v58 = *(_QWORD **)(a2 + 32);
        v104 = 0x200000000LL;
        v59 = *(_QWORD *)(a2 + 40);
        v103 = (__int64 *)v105;
        v93 = &v58[v59];
        if ( v58 == v93 )
          goto LABEL_85;
        v99 = 0;
        v60 = v58;
        do
        {
          v61 = *v60;
          v64 = sub_1EF3D60(a1, *v60, v7);
          v65 = (unsigned int)v104;
          if ( (unsigned int)v104 >= HIDWORD(v104) )
          {
            v86 = v64;
            sub_16CD150((__int64)&v103, v105, 0, 8, v62, v63);
            v65 = (unsigned int)v104;
            v64 = v86;
          }
          v7 = v103;
          v103[v65] = v64;
          v40 = v103;
          LODWORD(v104) = v104 + 1;
          ++v60;
          v99 |= v103[(unsigned int)v104 - 1] != v61;
        }
        while ( v93 != v60 );
        result = a2;
        if ( v99 )
        {
          result = sub_147A3C0(*a1, &v103, a3, a4);
          v40 = v103;
        }
LABEL_64:
        if ( v40 == (__int64 *)v105 )
          goto LABEL_87;
        v101 = result;
        _libc_free((unsigned __int64)v40);
        v7 = a1[2];
        v6 = *((_DWORD *)a1 + 8);
        result = v101;
        break;
      case 0xA:
LABEL_16:
        v22 = *(__int64 **)(a2 - 8);
        result = a2;
        if ( a1[5] != v22 )
          break;
        result = sub_145CF80((__int64)*a1, *v22, 0, 0);
LABEL_87:
        v7 = a1[2];
        v6 = *((_DWORD *)a1 + 8);
        break;
      default:
        goto LABEL_106;
    }
    v13 = (__int64)(a1 + 1);
    if ( v6 )
    {
      v8 = v6 - 1;
LABEL_19:
      v23 = 1;
      v19 = 0;
      v24 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v25 = &v7[2 * v24];
      v26 = *v25;
      if ( *v25 == a2 )
        return v25[1];
      while ( v26 != -8 )
      {
        if ( !v19 && v26 == -16 )
          v19 = v25;
        v24 = v8 & (v23 + v24);
        v25 = &v7[2 * v24];
        v26 = *v25;
        if ( *v25 == a2 )
          return v25[1];
        ++v23;
      }
      v74 = *((_DWORD *)a1 + 6);
      if ( !v19 )
        v19 = v25;
      a1[1] = (_QWORD *)((char *)a1[1] + 1);
      v18 = v74 + 1;
      if ( 4 * v18 < 3 * v6 )
      {
        if ( v6 - (v18 + *((_DWORD *)a1 + 7)) > v6 >> 3 )
        {
LABEL_10:
          *((_DWORD *)a1 + 6) = v18;
          if ( *v19 != -8 )
            --*((_DWORD *)a1 + 7);
          *v19 = a2;
          v19[1] = result;
          return result;
        }
        v102 = result;
        sub_1466670(v13, v6);
        v75 = *((_DWORD *)a1 + 8);
        if ( v75 )
        {
          v76 = v75 - 1;
          v77 = a1[2];
          v78 = 0;
          v79 = (v75 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v80 = 1;
          v18 = *((_DWORD *)a1 + 6) + 1;
          result = v102;
          v19 = &v77[2 * v79];
          v81 = *v19;
          if ( *v19 != a2 )
          {
            while ( v81 != -8 )
            {
              if ( !v78 && v81 == -16 )
                v78 = v19;
              v79 = v76 & (v80 + v79);
              v19 = &v77[2 * v79];
              v81 = *v19;
              if ( *v19 == a2 )
                goto LABEL_10;
              ++v80;
            }
            if ( v78 )
              v19 = v78;
          }
          goto LABEL_10;
        }
LABEL_106:
        JUMPOUT(0x4229BA);
      }
    }
    else
    {
LABEL_7:
      a1[1] = (_QWORD *)((char *)a1[1] + 1);
      v6 = 0;
    }
    v95 = result;
    sub_1466670(v13, 2 * v6);
    v14 = *((_DWORD *)a1 + 8);
    if ( v14 )
    {
      v15 = v14 - 1;
      v16 = a1[2];
      v17 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v18 = *((_DWORD *)a1 + 6) + 1;
      result = v95;
      v19 = &v16[2 * v17];
      v20 = *v19;
      if ( *v19 != a2 )
      {
        v83 = 1;
        v84 = 0;
        while ( v20 != -8 )
        {
          if ( !v84 && v20 == -16 )
            v84 = v19;
          v17 = v15 & (v83 + v17);
          v19 = &v16[2 * v17];
          v20 = *v19;
          if ( *v19 == a2 )
            goto LABEL_10;
          ++v83;
        }
        if ( v84 )
          v19 = v84;
      }
      goto LABEL_10;
    }
    goto LABEL_106;
  }
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = &v7[2 * v9];
  v11 = *v10;
  if ( *v10 != a2 )
  {
    for ( i = 1; ; i = v82 )
    {
      if ( v11 == -8 )
      {
        switch ( *(_WORD *)(a2 + 24) )
        {
          case 0:
          case 0xB:
LABEL_66:
            v13 = (__int64)(a1 + 1);
            result = a2;
            goto LABEL_19;
          case 1:
            goto LABEL_21;
          case 2:
            goto LABEL_23;
          case 3:
            goto LABEL_41;
          case 4:
            goto LABEL_34;
          case 5:
            goto LABEL_57;
          case 6:
            goto LABEL_25;
          case 7:
            goto LABEL_27;
          case 8:
            goto LABEL_43;
          case 9:
            goto LABEL_50;
          case 0xA:
            goto LABEL_16;
          default:
            goto LABEL_106;
        }
      }
      v82 = i + 1;
      v9 = v8 & (i + v9);
      v10 = &v7[2 * v9];
      v11 = *v10;
      if ( *v10 == a2 )
        break;
    }
  }
  if ( v10 == &v7[2 * v6] )
  {
    switch ( *(_WORD *)(a2 + 24) )
    {
      case 0:
      case 0xB:
        goto LABEL_66;
      case 1:
        goto LABEL_21;
      case 2:
        goto LABEL_23;
      case 3:
        goto LABEL_41;
      case 4:
        goto LABEL_34;
      case 5:
        goto LABEL_57;
      case 6:
        goto LABEL_25;
      case 7:
        goto LABEL_27;
      case 8:
        goto LABEL_43;
      case 9:
        goto LABEL_50;
      case 0xA:
        goto LABEL_16;
      default:
        goto LABEL_106;
    }
  }
  return v10[1];
}
