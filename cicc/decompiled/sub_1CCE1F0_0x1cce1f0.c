// Function: sub_1CCE1F0
// Address: 0x1cce1f0
//
__int64 __fastcall sub_1CCE1F0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // r12
  unsigned __int64 v16; // rdi
  _QWORD *v17; // r8
  _QWORD *v18; // r9
  int v19; // ebx
  unsigned __int64 v20; // r12
  __int64 v21; // r14
  _QWORD *v22; // r15
  unsigned int i; // r13d
  __int64 v24; // rax
  int v25; // eax
  __int64 *v26; // rbx
  __int64 v27; // r13
  unsigned int v28; // eax
  __int64 v29; // rcx
  __int64 v30; // r12
  __int64 v31; // rax
  unsigned int v32; // ecx
  _QWORD *v33; // rdx
  __int64 v34; // rax
  int v35; // r10d
  int v36; // edx
  int v37; // edi
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 j; // rax
  __int64 v41; // r13
  __int64 v42; // rcx
  unsigned int v43; // eax
  __int64 v44; // rsi
  __int64 v45; // r12
  unsigned __int64 v46; // rax
  int v47; // ebx
  unsigned __int64 v48; // r15
  __int64 v49; // rax
  int v50; // edx
  unsigned int v51; // ebx
  _QWORD *v52; // r12
  int v53; // r14d
  __int64 v54; // rax
  __int64 *v55; // r15
  __int64 v56; // rdi
  _QWORD *v57; // rax
  __int64 v58; // r15
  _QWORD *v59; // rdi
  _QWORD *v60; // rdi
  unsigned __int64 v61; // rsi
  _QWORD *v62; // rax
  __int64 v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // r12
  _QWORD *v66; // rbx
  _QWORD *v67; // r15
  __int64 v68; // rax
  unsigned __int64 v69; // r13
  char v70; // al
  __int64 v71; // rax
  double v72; // xmm4_8
  double v73; // xmm5_8
  unsigned int v74; // eax
  __int64 v75; // rsi
  int v76; // r10d
  _QWORD *v78; // rdi
  unsigned int v79; // r15d
  __int64 v80; // rcx
  __int64 v81; // [rsp+18h] [rbp-8B8h]
  __int64 **v82; // [rsp+28h] [rbp-8A8h]
  int v84; // [rsp+38h] [rbp-898h]
  __int64 *v85; // [rsp+38h] [rbp-898h]
  unsigned __int8 v86; // [rsp+40h] [rbp-890h]
  __int64 v87; // [rsp+40h] [rbp-890h]
  __int64 v88; // [rsp+48h] [rbp-888h]
  __int64 *v89; // [rsp+48h] [rbp-888h]
  __int64 v90; // [rsp+48h] [rbp-888h]
  __int64 **v91; // [rsp+50h] [rbp-880h]
  __int64 v92; // [rsp+58h] [rbp-878h]
  __int64 v93; // [rsp+58h] [rbp-878h]
  __int64 *v94; // [rsp+58h] [rbp-878h]
  __int64 v95; // [rsp+60h] [rbp-870h] BYREF
  __int64 v96; // [rsp+68h] [rbp-868h]
  __int64 v97; // [rsp+70h] [rbp-860h]
  __int64 v98; // [rsp+78h] [rbp-858h]
  _QWORD *v99; // [rsp+80h] [rbp-850h] BYREF
  unsigned int v100; // [rsp+88h] [rbp-848h]
  unsigned int v101; // [rsp+8Ch] [rbp-844h]
  _QWORD v102[128]; // [rsp+90h] [rbp-840h] BYREF
  __int64 *v103; // [rsp+490h] [rbp-440h] BYREF
  __int64 v104; // [rsp+498h] [rbp-438h]
  _BYTE v105[1072]; // [rsp+4A0h] [rbp-430h] BYREF

  v82 = *(__int64 ***)(a2 + 24);
  if ( *(__int64 ***)(a2 + 16) != v82 )
  {
    v91 = *(__int64 ***)(a2 + 16);
    v86 = 0;
LABEL_4:
    v10 = **v91;
    v88 = v10;
    if ( !v10 || sub_15E4F60(v10) )
      goto LABEL_3;
    v101 = 128;
    v11 = *(_QWORD *)(v88 + 80);
    v99 = v102;
    v95 = 0;
    v12 = v11 - 24;
    v96 = 0;
    v97 = 0;
    if ( !v11 )
      v12 = 0;
    v13 = v102;
    v98 = 0;
    v100 = 1;
    v102[0] = v12;
    v14 = 1;
    while ( 1 )
    {
      v15 = v13[v14 - 1];
      v100 = v14 - 1;
      v16 = sub_157EBA0(v15);
      if ( v16 )
      {
        v19 = sub_15F4D60(v16);
        v20 = sub_157EBA0(v15);
        if ( (unsigned __int64)v19 > 0xFFFFFFFFFFFFFFFLL )
LABEL_135:
          sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
        v92 = 8LL * v19;
        if ( v19 )
        {
          v21 = sub_22077B0(8LL * v19);
          v22 = (_QWORD *)v21;
          for ( i = 0; i != v19; ++i )
          {
            v24 = sub_15F4DF0(v20, i);
            if ( v22 )
              *v22 = v24;
            ++v22;
          }
          v25 = v19;
          v26 = (__int64 *)v21;
          v27 = v21 + 8LL * (unsigned int)(v25 - 1) + 8;
          while ( 1 )
          {
            v30 = *v26;
            if ( (_DWORD)v98 )
            {
              v28 = (v98 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
              v29 = *(_QWORD *)(v96 + 8LL * v28);
              if ( v30 == v29 )
                goto LABEL_18;
              v37 = 1;
              while ( v29 != -8 )
              {
                LODWORD(v17) = v37 + 1;
                v28 = (v98 - 1) & (v37 + v28);
                v29 = *(_QWORD *)(v96 + 8LL * v28);
                if ( v30 == v29 )
                  goto LABEL_18;
                ++v37;
              }
            }
            v31 = v100;
            if ( v100 >= v101 )
            {
              sub_16CD150((__int64)&v99, v102, 0, 8, (int)v17, (int)v18);
              v31 = v100;
            }
            v99[v31] = v30;
            ++v100;
            if ( !(_DWORD)v98 )
            {
              ++v95;
              goto LABEL_96;
            }
            LODWORD(v17) = v98 - 1;
            v32 = (v98 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
            v33 = (_QWORD *)(v96 + 8LL * v32);
            v34 = *v33;
            if ( v30 == *v33 )
            {
LABEL_18:
              if ( (__int64 *)v27 == ++v26 )
                goto LABEL_33;
            }
            else
            {
              v35 = 1;
              v18 = 0;
              while ( v34 != -8 )
              {
                if ( v18 || v34 != -16 )
                  v33 = v18;
                LODWORD(v18) = v35 + 1;
                v32 = (unsigned int)v17 & (v35 + v32);
                v34 = *(_QWORD *)(v96 + 8LL * v32);
                if ( v30 == v34 )
                  goto LABEL_18;
                ++v35;
                v18 = v33;
                v33 = (_QWORD *)(v96 + 8LL * v32);
              }
              if ( !v18 )
                v18 = v33;
              ++v95;
              v36 = v97 + 1;
              if ( 4 * ((int)v97 + 1) < (unsigned int)(3 * v98) )
              {
                if ( (int)v98 - HIDWORD(v97) - v36 <= (unsigned int)v98 >> 3 )
                {
                  sub_13B3D40((__int64)&v95, v98);
                  if ( !(_DWORD)v98 )
                  {
LABEL_136:
                    LODWORD(v97) = v97 + 1;
                    BUG();
                  }
                  LODWORD(v17) = 1;
                  v78 = 0;
                  v79 = (v98 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
                  v18 = (_QWORD *)(v96 + 8LL * v79);
                  v80 = *v18;
                  v36 = v97 + 1;
                  if ( v30 != *v18 )
                  {
                    while ( v80 != -8 )
                    {
                      if ( v80 == -16 && !v78 )
                        v78 = v18;
                      v79 = (v98 - 1) & ((_DWORD)v17 + v79);
                      v18 = (_QWORD *)(v96 + 8LL * v79);
                      v80 = *v18;
                      if ( v30 == *v18 )
                        goto LABEL_30;
                      LODWORD(v17) = (_DWORD)v17 + 1;
                    }
                    if ( v78 )
                      v18 = v78;
                  }
                }
                goto LABEL_30;
              }
LABEL_96:
              sub_13B3D40((__int64)&v95, 2 * v98);
              if ( !(_DWORD)v98 )
                goto LABEL_136;
              v74 = (v98 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
              v18 = (_QWORD *)(v96 + 8LL * v74);
              v75 = *v18;
              v36 = v97 + 1;
              if ( v30 != *v18 )
              {
                v76 = 1;
                v17 = 0;
                while ( v75 != -8 )
                {
                  if ( !v17 && v75 == -16 )
                    v17 = v18;
                  v74 = (v98 - 1) & (v76 + v74);
                  v18 = (_QWORD *)(v96 + 8LL * v74);
                  v75 = *v18;
                  if ( v30 == *v18 )
                    goto LABEL_30;
                  ++v76;
                }
                if ( v17 )
                  v18 = v17;
              }
LABEL_30:
              LODWORD(v97) = v36;
              if ( *v18 != -8 )
                --HIDWORD(v97);
              ++v26;
              *v18 = v30;
              if ( (__int64 *)v27 == v26 )
              {
LABEL_33:
                if ( v21 )
                  j_j___libc_free_0(v21, v92);
                break;
              }
            }
          }
        }
      }
      v14 = v100;
      if ( !v100 )
        break;
      v13 = v99;
    }
    v38 = 0;
    v39 = *(_QWORD *)(v88 + 80);
    v93 = v88 + 72;
    for ( j = v39; v88 + 72 != j; ++v38 )
      j = *(_QWORD *)(j + 8);
    if ( (unsigned int)v97 == v38 )
    {
      j___libc_free_0(v96);
      if ( v99 != v102 )
        _libc_free((unsigned __int64)v99);
      goto LABEL_3;
    }
    v103 = (__int64 *)v105;
    v104 = 0x8000000000LL;
    v41 = *(_QWORD *)(v39 + 8);
    if ( v93 == v41 )
      goto LABEL_91;
    v42 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v45 = v41 - 24;
        if ( !v41 )
          v45 = 0;
        if ( (_DWORD)v98 )
          break;
        if ( HIDWORD(v104) <= (unsigned int)v42 )
          goto LABEL_116;
LABEL_52:
        v103[v42] = v45;
        LODWORD(v104) = v104 + 1;
        v46 = sub_157EBA0(v45);
        v89 = (__int64 *)v46;
        if ( v46 )
        {
          v47 = sub_15F4D60(v46);
          v84 = v47;
          v48 = sub_157EBA0(v45);
          if ( (unsigned __int64)v47 > 0xFFFFFFFFFFFFFFFLL )
            goto LABEL_135;
          v87 = 8LL * v47;
          if ( v47 )
          {
            v49 = sub_22077B0(8LL * v47);
            v50 = v47;
            v89 = (__int64 *)v49;
            v81 = v45;
            v51 = 0;
            v52 = (_QWORD *)v49;
            v53 = v50;
            do
            {
              v54 = sub_15F4DF0(v48, v51);
              if ( v52 )
                *v52 = v54;
              ++v51;
              ++v52;
            }
            while ( v53 != v51 );
            v45 = v81;
            if ( v84 )
            {
              v55 = v89;
              do
              {
                v56 = *v55++;
                sub_157F2D0(v56, v81, 0);
              }
              while ( &v89[(unsigned int)(v84 - 1) + 1] != v55 );
            }
          }
          else
          {
            v89 = 0;
          }
        }
        else
        {
          v87 = 0;
        }
        v57 = (_QWORD *)sub_157EBA0(v45);
        sub_15F20C0(v57);
        v58 = sub_157E9C0(v45);
        v59 = sub_1648A60(56, 0);
        if ( v59 )
          sub_15F82E0((__int64)v59, v58, v45);
        if ( v89 )
          j_j___libc_free_0(v89, v87);
        v42 = (unsigned int)v104;
        v41 = *(_QWORD *)(v41 + 8);
        if ( v93 == v41 )
        {
LABEL_67:
          v94 = v103;
          v85 = &v103[v42];
          if ( v103 != v85 )
          {
            do
            {
              v60 = (_QWORD *)(a1 + 16);
              v90 = *v94;
              v61 = *(_QWORD *)(*v94 + 56);
              v62 = *(_QWORD **)(a1 + 24);
              if ( v62 )
              {
                do
                {
                  while ( 1 )
                  {
                    v63 = v62[2];
                    v64 = v62[3];
                    if ( v62[4] >= v61 )
                      break;
                    v62 = (_QWORD *)v62[3];
                    if ( !v64 )
                      goto LABEL_73;
                  }
                  v60 = v62;
                  v62 = (_QWORD *)v62[2];
                }
                while ( v63 );
LABEL_73:
                if ( v60 != (_QWORD *)(a1 + 16) && v60[4] > v61 )
                  v60 = (_QWORD *)(a1 + 16);
              }
              v65 = v60[5];
              v66 = *(_QWORD **)(v90 + 48);
              v67 = (_QWORD *)(v90 + 40);
LABEL_81:
              if ( v67 != v66 )
              {
                do
                {
                  v69 = *v67 & 0xFFFFFFFFFFFFFFF8LL;
                  v67 = (_QWORD *)v69;
                  if ( !v69 )
                    BUG();
                  v70 = *(_BYTE *)(v69 - 8);
                  if ( v70 == 78 )
                  {
                    v68 = *(_QWORD *)(v69 - 48);
                    if ( *(_BYTE *)(v68 + 16) || (*(_BYTE *)(v68 + 33) & 0x20) == 0 )
                      sub_13983A0(v65, (v69 - 24) | 4);
                  }
                  else if ( v70 == 29 )
                  {
                    sub_13983A0(v65, v69 - 24);
                    if ( !*(_QWORD *)(v69 - 16) )
                      goto LABEL_81;
                    goto LABEL_86;
                  }
                  if ( !*(_QWORD *)(v69 - 16) )
                    goto LABEL_81;
LABEL_86:
                  v71 = sub_1599EF0(*(__int64 ***)(v69 - 24));
                  sub_164D160(v69 - 24, v71, a3, a4, a5, a6, v72, v73, a9, a10);
                }
                while ( (_QWORD *)v69 != v66 );
              }
              sub_157F980(v90);
              ++v94;
            }
            while ( v94 != v85 );
            v85 = v103;
          }
          if ( v85 != (__int64 *)v105 )
            _libc_free((unsigned __int64)v85);
LABEL_91:
          j___libc_free_0(v96);
          if ( v99 != v102 )
            _libc_free((unsigned __int64)v99);
          v86 = 1;
LABEL_3:
          if ( v82 == ++v91 )
            return v86;
          goto LABEL_4;
        }
      }
      LODWORD(v17) = 1;
      v43 = (v98 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
      v44 = *(_QWORD *)(v96 + 8LL * v43);
      if ( v45 != v44 )
      {
        while ( v44 != -8 )
        {
          LODWORD(v18) = (_DWORD)v17 + 1;
          v43 = (v98 - 1) & ((_DWORD)v17 + v43);
          v44 = *(_QWORD *)(v96 + 8LL * v43);
          if ( v45 == v44 )
            goto LABEL_47;
          LODWORD(v17) = (_DWORD)v17 + 1;
        }
        if ( HIDWORD(v104) > (unsigned int)v42 )
          goto LABEL_52;
LABEL_116:
        sub_16CD150((__int64)&v103, v105, 0, 8, (int)v17, (int)v18);
        v42 = (unsigned int)v104;
        goto LABEL_52;
      }
LABEL_47:
      v41 = *(_QWORD *)(v41 + 8);
      if ( v93 == v41 )
        goto LABEL_67;
    }
  }
  return 0;
}
