// Function: sub_1A980B0
// Address: 0x1a980b0
//
void __fastcall sub_1A980B0(_QWORD *a1, unsigned __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rdx
  _QWORD *v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // r15
  unsigned int v17; // esi
  __int64 v18; // rbx
  unsigned int v19; // edi
  __int64 v20; // r8
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r13
  __int64 v25; // r10
  __int64 *v26; // r12
  __int64 v27; // r15
  __int64 *v28; // rcx
  __int64 *v29; // rdx
  int v30; // esi
  __int64 v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // r10
  __int64 v34; // rbx
  __int64 *v35; // rsi
  __int64 v36; // rdi
  __int64 v37; // rdx
  int v38; // r8d
  __int64 v39; // rcx
  char v40; // al
  __int64 v41; // rdx
  int v42; // r12d
  __int64 v43; // rdi
  __int64 *v44; // r12
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rsi
  __int64 v48; // rsi
  unsigned __int8 *v49; // rsi
  __int64 v50; // rcx
  _QWORD *v51; // rax
  __int64 v52; // r9
  __int64 v53; // r13
  int v54; // r10d
  int v55; // r10d
  __int64 *v56; // r13
  int v57; // edx
  __int64 v58; // rcx
  __int64 v59; // rax
  __int64 *v60; // rax
  __int64 *v61; // r10
  int v62; // eax
  int v63; // r11d
  __int64 *v64; // r9
  __int64 *v65; // rbx
  __int64 v66; // [rsp+0h] [rbp-170h]
  __int64 v67; // [rsp+8h] [rbp-168h]
  _QWORD *v68; // [rsp+10h] [rbp-160h]
  __int64 *v69; // [rsp+18h] [rbp-158h]
  __int64 v70; // [rsp+30h] [rbp-140h]
  __int64 *v74; // [rsp+60h] [rbp-110h]
  __int64 v75; // [rsp+60h] [rbp-110h]
  __int64 v76; // [rsp+68h] [rbp-108h]
  __int64 v77; // [rsp+68h] [rbp-108h]
  __int64 v78; // [rsp+68h] [rbp-108h]
  __int64 v79; // [rsp+68h] [rbp-108h]
  unsigned int v82; // [rsp+8Ch] [rbp-E4h]
  __int64 v83; // [rsp+98h] [rbp-D8h] BYREF
  __int64 v84[4]; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 *v85; // [rsp+C0h] [rbp-B0h] BYREF
  __int16 v86; // [rsp+D0h] [rbp-A0h]
  _QWORD v87[2]; // [rsp+E0h] [rbp-90h] BYREF
  __int16 v88; // [rsp+F0h] [rbp-80h]
  __int64 v89; // [rsp+100h] [rbp-70h] BYREF
  __int64 v90; // [rsp+108h] [rbp-68h]
  __int64 v91; // [rsp+110h] [rbp-60h]
  unsigned int v92; // [rsp+118h] [rbp-58h]
  __int64 v93[2]; // [rsp+120h] [rbp-50h] BYREF
  __int64 v94; // [rsp+130h] [rbp-40h] BYREF

  if ( a2 )
  {
    v89 = 0;
    v69 = (__int64 *)sub_15F2050(a5);
    v67 = (__int64)(8 * a2) >> 3;
    v90 = 0;
    v91 = 0;
    v92 = 0;
    v82 = 0;
    v66 = (__int64)(8 * a2 - ((8 * a2) & 0xFFFFFFFFFFFFFFE0LL)) >> 3;
    v68 = (_QWORD *)((char *)a1 + ((8 * a2) & 0xFFFFFFFFFFFFFFE0LL));
    v70 = (__int64)(8 * a2) >> 5;
    v7 = 0;
    while ( 1 )
    {
      v8 = v7;
      v9 = *(_QWORD *)(a4 + 8 * v7);
      if ( v70 > 0 )
      {
        v10 = a1;
        while ( v9 != *v10 )
        {
          if ( v9 == v10[1] )
          {
            v11 = v10 + 1 - a1;
            goto LABEL_11;
          }
          if ( v9 == v10[2] )
          {
            v11 = v10 + 2 - a1;
            goto LABEL_11;
          }
          if ( v9 == v10[3] )
          {
            v11 = v10 + 3 - a1;
            goto LABEL_11;
          }
          v10 += 4;
          if ( v68 == v10 )
          {
            v50 = v66;
            goto LABEL_45;
          }
        }
LABEL_10:
        LODWORD(v11) = v10 - a1;
        goto LABEL_11;
      }
      v50 = v67;
      v10 = a1;
LABEL_45:
      if ( v50 == 2 )
        goto LABEL_51;
      if ( v50 == 3 )
        break;
      if ( v50 != 1 )
      {
        LODWORD(v11) = v67;
        goto LABEL_11;
      }
LABEL_53:
      LODWORD(v11) = v67;
      if ( v9 == *v10 )
        goto LABEL_10;
LABEL_11:
      v12 = sub_1643350(*(_QWORD **)(a6 + 24));
      v13 = sub_159C470(v12, (unsigned int)(a3 + v11), 0);
      v14 = sub_1643350(*(_QWORD **)(a6 + 24));
      v15 = sub_159C470(v14, v82 + a3, 0);
      v16 = &a1[v8];
      v17 = v92;
      v76 = v15;
      v18 = *(_QWORD *)*v16;
      v83 = v18;
      if ( !v92 )
      {
        ++v89;
LABEL_89:
        v17 = 2 * v92;
LABEL_90:
        sub_17B2740((__int64)&v89, v17);
        sub_1A971D0((__int64)&v89, &v83, v93);
        v56 = (__int64 *)v93[0];
        v58 = v83;
        v57 = v91 + 1;
        goto LABEL_65;
      }
      v19 = v92 - 1;
      v20 = v90;
      v21 = (v92 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v22 = (__int64 *)(v90 + 16LL * v21);
      v23 = *v22;
      if ( v18 == *v22 )
        goto LABEL_13;
      v52 = *v22;
      LODWORD(v53) = (v92 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v54 = 1;
      while ( v52 != -8 )
      {
        v53 = v19 & ((_DWORD)v53 + v54);
        v52 = *(_QWORD *)(v90 + 16 * v53);
        if ( v18 == v52 )
          goto LABEL_81;
        ++v54;
      }
      v55 = 1;
      v56 = 0;
      while ( v23 != -8 )
      {
        if ( v23 != -16 || v56 )
          v22 = v56;
        v21 = v19 & (v55 + v21);
        v56 = (__int64 *)(v90 + 16LL * v21);
        v23 = *v56;
        if ( v18 == *v56 )
          goto LABEL_68;
        ++v55;
        v64 = v22;
        v22 = (__int64 *)(v90 + 16LL * v21);
        v56 = v64;
      }
      if ( !v56 )
        v56 = v22;
      ++v89;
      v57 = v91 + 1;
      if ( 4 * ((int)v91 + 1) >= 3 * v92 )
        goto LABEL_89;
      v58 = v18;
      if ( v92 - HIDWORD(v91) - v57 <= v92 >> 3 )
        goto LABEL_90;
LABEL_65:
      LODWORD(v91) = v57;
      if ( *v56 != -8 )
        --HIDWORD(v91);
      *v56 = v58;
      v56[1] = 0;
LABEL_68:
      v59 = v18;
      if ( *(_BYTE *)(v18 + 8) == 16 )
      {
        v59 = **(_QWORD **)(v18 + 16);
        if ( *(_BYTE *)(v59 + 8) == 16 )
          v59 = **(_QWORD **)(v59 + 16);
      }
      v60 = (__int64 *)sub_16471D0((_QWORD *)*v69, *(_DWORD *)(v59 + 8) >> 8);
      if ( *(_BYTE *)(v18 + 8) == 16 )
        v60 = sub_16463B0(v60, *(_QWORD *)(v18 + 32));
      v93[0] = (__int64)v60;
      v56[1] = sub_15E26F0(v69, 76, v93, 1);
      v17 = v92;
      if ( !v92 )
      {
        ++v89;
LABEL_73:
        v17 *= 2;
        goto LABEL_74;
      }
      v52 = v83;
      v19 = v92 - 1;
      v20 = v90;
      v21 = (v92 - 1) & (((unsigned int)v83 >> 9) ^ ((unsigned int)v83 >> 4));
      v22 = (__int64 *)(v90 + 16LL * v21);
      v23 = *v22;
      if ( v83 == *v22 )
      {
LABEL_13:
        v24 = v22[1];
        goto LABEL_14;
      }
LABEL_81:
      v63 = 1;
      v61 = 0;
      while ( v23 != -8 )
      {
        if ( v23 != -16 || v61 )
          v22 = v61;
        v21 = v19 & (v63 + v21);
        v65 = (__int64 *)(v20 + 16LL * v21);
        v23 = *v65;
        if ( v52 == *v65 )
        {
          v24 = v65[1];
          goto LABEL_14;
        }
        ++v63;
        v61 = v22;
        v22 = (__int64 *)(v20 + 16LL * v21);
      }
      if ( !v61 )
        v61 = v22;
      ++v89;
      v62 = v91 + 1;
      if ( 4 * ((int)v91 + 1) >= 3 * v17 )
        goto LABEL_73;
      if ( v17 - (v62 + HIDWORD(v91)) > v17 >> 3 )
        goto LABEL_75;
LABEL_74:
      sub_17B2740((__int64)&v89, v17);
      sub_1A971D0((__int64)&v89, &v83, v93);
      v61 = (__int64 *)v93[0];
      v52 = v83;
      v62 = v91 + 1;
LABEL_75:
      LODWORD(v91) = v62;
      if ( *v61 != -8 )
        --HIDWORD(v91);
      *v61 = v52;
      v24 = 0;
      v61[1] = 0;
LABEL_14:
      sub_1A956E0(v93, *v16, (__int64)".relocated", 10, byte_3F871B3, 0);
      v84[1] = v13;
      v86 = 260;
      v25 = *(_QWORD *)(a6 + 56);
      v84[0] = a5;
      v85 = v93;
      v26 = *(__int64 **)(a6 + 48);
      v84[2] = v76;
      v27 = *(_QWORD *)(*(_QWORD *)v24 + 24LL);
      v88 = 257;
      v28 = &v26[7 * v25];
      if ( v28 == v26 )
      {
        v79 = v25;
        v51 = sub_1648AB0(72, 4u, 16 * (int)v25);
        v33 = v79;
        v34 = (__int64)v51;
        if ( !v51 )
          goto LABEL_105;
        v78 = (__int64)v51;
        v39 = -96;
        v38 = 4;
      }
      else
      {
        v29 = v26;
        v30 = 0;
        do
        {
          v31 = v29[5] - v29[4];
          v29 += 7;
          v30 += v31 >> 3;
        }
        while ( v28 != v29 );
        v74 = &v26[7 * v25];
        v77 = v25;
        v32 = sub_1648AB0(72, v30 + 4, 16 * (int)v25);
        v33 = v77;
        v34 = (__int64)v32;
        if ( !v32 )
        {
LABEL_105:
          v78 = 0;
          v34 = 0;
          goto LABEL_22;
        }
        v78 = (__int64)v32;
        v35 = v26;
        LODWORD(v36) = 0;
        do
        {
          v37 = v35[5] - v35[4];
          v35 += 7;
          v36 = (unsigned int)(v37 >> 3) + (unsigned int)v36;
        }
        while ( v74 != v35 );
        v38 = v36 + 4;
        v39 = -24 - 8 * (3 * v36 + 9);
      }
      v75 = v33;
      sub_15F1EA0(v34, **(_QWORD **)(v27 + 16), 54, v34 + v39, v38, 0);
      *(_QWORD *)(v34 + 56) = 0;
      sub_15F5B40(v34, v27, v24, v84, 3, (__int64)v87, v26, v75);
LABEL_22:
      v40 = *(_BYTE *)(*(_QWORD *)v34 + 8LL);
      if ( v40 == 16 )
        v40 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)v34 + 16LL) + 8LL);
      if ( (unsigned __int8)(v40 - 1) <= 5u || *(_BYTE *)(v34 + 16) == 76 )
      {
        v41 = *(_QWORD *)(a6 + 32);
        v42 = *(_DWORD *)(a6 + 40);
        if ( v41 )
          sub_1625C10(v34, 3, v41);
        sub_15F2440(v34, v42);
      }
      v43 = *(_QWORD *)(a6 + 8);
      if ( v43 )
      {
        v44 = *(__int64 **)(a6 + 16);
        sub_157E9D0(v43 + 40, v34);
        v45 = *(_QWORD *)(v34 + 24);
        v46 = *v44;
        *(_QWORD *)(v34 + 32) = v44;
        v46 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v34 + 24) = v46 | v45 & 7;
        *(_QWORD *)(v46 + 8) = v34 + 24;
        *v44 = *v44 & 7 | (v34 + 24);
      }
      sub_164B780(v78, (__int64 *)&v85);
      v47 = *(_QWORD *)a6;
      if ( *(_QWORD *)a6 )
      {
        v87[0] = *(_QWORD *)a6;
        sub_1623A60((__int64)v87, v47, 2);
        v48 = *(_QWORD *)(v34 + 48);
        if ( v48 )
          sub_161E7C0(v34 + 48, v48);
        v49 = (unsigned __int8 *)v87[0];
        *(_QWORD *)(v34 + 48) = v87[0];
        if ( v49 )
          sub_1623210((__int64)v87, v49, v34 + 48);
      }
      if ( (__int64 *)v93[0] != &v94 )
        j_j___libc_free_0(v93[0], v94 + 1);
      ++v82;
      *(_WORD *)(v34 + 18) = *(_WORD *)(v34 + 18) & 0x8003 | 0x24;
      v7 = v82;
      if ( a2 <= v82 )
      {
        j___libc_free_0(v90);
        return;
      }
    }
    if ( v9 == *v10 )
      goto LABEL_10;
    ++v10;
LABEL_51:
    if ( v9 == *v10 )
      goto LABEL_10;
    ++v10;
    goto LABEL_53;
  }
}
