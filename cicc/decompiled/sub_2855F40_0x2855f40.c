// Function: sub_2855F40
// Address: 0x2855f40
//
void __fastcall sub_2855F40(__int64 a1)
{
  __int64 v2; // rcx
  __int64 v3; // rdi
  __int64 v4; // rdi
  unsigned __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 *v7; // r12
  __int64 v8; // rbx
  __int64 *v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // rbx
  bool v12; // r12
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rax
  __int64 v18; // r14
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // r12
  char v22; // r11
  __int64 v23; // r13
  __int64 v24; // r8
  _QWORD *v25; // rdi
  _QWORD *v26; // rsi
  __int64 *v27; // r8
  __int64 *v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned int v32; // eax
  unsigned int v33; // ecx
  unsigned int v34; // edx
  __int64 *v35; // rax
  __int64 v36; // rsi
  unsigned __int64 v37; // rdi
  unsigned int v38; // r14d
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // rsi
  unsigned __int64 v42; // rsi
  __int64 v45; // rax
  __int64 *v46; // r13
  unsigned __int64 v47; // rcx
  int v48; // ecx
  __int64 v49; // r9
  unsigned int v50; // edx
  __int64 v51; // r8
  __int64 v52; // rdi
  int v54; // edx
  __int64 *v56; // rbx
  __int64 v57; // r13
  __int64 v58; // rdi
  unsigned int v59; // ecx
  __int64 v60; // rsi
  unsigned int v61; // edx
  __int64 *v62; // rax
  __int64 v63; // r8
  unsigned __int64 v64; // rax
  int v65; // eax
  __int64 *v66; // rbx
  __int64 v67; // r14
  unsigned int v68; // r13d
  __int64 v69; // rdi
  __int64 *v70; // rax
  int v71; // eax
  int v72; // eax
  int v73; // edi
  int v74; // edi
  int v75; // r9d
  unsigned __int64 v76; // rcx
  __int64 v77; // [rsp+0h] [rbp-D0h]
  __int64 v78; // [rsp+8h] [rbp-C8h]
  __int64 v79; // [rsp+8h] [rbp-C8h]
  __int64 v80; // [rsp+18h] [rbp-B8h]
  unsigned int v81; // [rsp+20h] [rbp-B0h]
  __int64 v82; // [rsp+20h] [rbp-B0h]
  unsigned int v83; // [rsp+28h] [rbp-A8h]
  __int64 v84; // [rsp+28h] [rbp-A8h]
  __int64 v85; // [rsp+30h] [rbp-A0h]
  __int64 v86; // [rsp+30h] [rbp-A0h]
  __int64 v87; // [rsp+30h] [rbp-A0h]
  __int64 *v88; // [rsp+38h] [rbp-98h]
  __int64 v89; // [rsp+38h] [rbp-98h]
  unsigned __int64 v90; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v91; // [rsp+48h] [rbp-88h]
  char v92; // [rsp+50h] [rbp-80h]
  __int64 v93; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v94; // [rsp+68h] [rbp-68h]
  __int64 v95; // [rsp+70h] [rbp-60h]
  int v96; // [rsp+78h] [rbp-58h]
  char v97; // [rsp+7Ch] [rbp-54h]
  char v98; // [rsp+80h] [rbp-50h] BYREF

  v94 = (__int64 *)&v98;
  v2 = *(_QWORD *)(a1 + 1320);
  v93 = 0;
  v3 = *(unsigned int *)(a1 + 1328);
  v95 = 4;
  v96 = 0;
  v97 = 1;
  while ( 1 )
  {
LABEL_2:
    v4 = v2 + 2184 * v3;
    if ( v2 == v4 )
    {
      if ( (unsigned int)qword_5001308 > 1 )
        break;
    }
    else
    {
      v5 = 1;
      while ( 1 )
      {
        v6 = *(unsigned int *)(v2 + 768);
        if ( (unsigned int)v6 >= (unsigned int)qword_5001308 )
          break;
        v5 *= v6;
        if ( (unsigned int)qword_5001308 <= v5 )
          break;
        v2 += 2184;
        if ( v4 == v2 )
          goto LABEL_103;
      }
    }
    v7 = *(__int64 **)(a1 + 36312);
    v88 = &v7[*(unsigned int *)(a1 + 36320)];
    if ( v7 != v88 )
    {
      v83 = 0;
      v85 = 0;
      while ( 1 )
      {
        v8 = *v7;
        if ( (unsigned __int8)sub_DF9980(*(_QWORD *)(a1 + 48)) && *(_WORD *)(v8 + 24) != 8 )
          goto LABEL_16;
        if ( v97 )
          break;
        if ( sub_C8CA60((__int64)&v93, v8) )
        {
LABEL_16:
          if ( v88 == ++v7 )
            goto LABEL_17;
        }
        else
        {
LABEL_42:
          if ( (unsigned __int8)sub_DF9980(*(_QWORD *)(a1 + 48)) && !*(_WORD *)(**(_QWORD **)(v8 + 32) + 24LL) )
          {
            if ( !v97 )
              goto LABEL_117;
            v70 = v94;
            v29 = HIDWORD(v95);
            v28 = &v94[HIDWORD(v95)];
            if ( v94 != v28 )
            {
              while ( v8 != *v70 )
              {
                if ( v28 == ++v70 )
                  goto LABEL_116;
              }
              goto LABEL_16;
            }
LABEL_116:
            if ( HIDWORD(v95) < (unsigned int)v95 )
            {
              ++v7;
              ++HIDWORD(v95);
              *v28 = v8;
              ++v93;
              if ( v88 == v7 )
              {
LABEL_17:
                v11 = v85;
                v12 = v85 == 0;
                goto LABEL_18;
              }
            }
            else
            {
LABEL_117:
              ++v7;
              sub_C8CC70((__int64)&v93, v8, (__int64)v28, v29, v30, v31);
              if ( v88 == v7 )
                goto LABEL_17;
            }
          }
          else if ( v85 )
          {
            v32 = *(_DWORD *)(a1 + 36304);
            v80 = *(_QWORD *)(a1 + 36288);
            v81 = v32;
            if ( v32 )
            {
              v33 = v32 - 1;
              v34 = (v32 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
              v35 = (__int64 *)(*(_QWORD *)(a1 + 36288) + 16LL * v34);
              v36 = *v35;
              if ( v8 == *v35 )
                goto LABEL_47;
              v65 = 1;
              while ( v36 != -4096 )
              {
                v74 = v65 + 1;
                v34 = v33 & (v65 + v34);
                v35 = (__int64 *)(v80 + 16LL * v34);
                v36 = *v35;
                if ( v8 == *v35 )
                  goto LABEL_47;
                v65 = v74;
              }
            }
            v35 = (__int64 *)(v80 + 16LL * v81);
LABEL_47:
            v37 = v35[1];
            if ( (v37 & 1) != 0 )
            {
              v38 = sub_39FAC40(~(-1LL << (v37 >> 58)) & (v37 >> 1));
            }
            else
            {
              if ( *(_QWORD *)v37 == *(_QWORD *)v37 + 8LL * *(unsigned int *)(v37 + 8) )
              {
                v38 = 0;
                goto LABEL_71;
              }
              v79 = v8;
              v38 = 0;
              v56 = *(__int64 **)v37;
              v57 = *(_QWORD *)v37 + 8LL * *(unsigned int *)(v37 + 8);
              do
              {
                v58 = *v56++;
                v38 += sub_39FAC40(v58);
              }
              while ( v56 != (__int64 *)v57 );
              v8 = v79;
            }
            if ( v83 < v38 )
            {
              v85 = v8;
              goto LABEL_51;
            }
LABEL_71:
            if ( v38 == v83 )
            {
LABEL_51:
              if ( v81 )
              {
                v39 = (v81 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
                v40 = (__int64 *)(v80 + 16LL * v39);
                v41 = *v40;
                if ( v8 == *v40 )
                {
LABEL_53:
                  v42 = v40[1];
                  if ( (v42 & 1) == 0 )
                    goto LABEL_75;
                  goto LABEL_54;
                }
                v72 = 1;
                while ( v41 != -4096 )
                {
                  v73 = v72 + 1;
                  v39 = (v81 - 1) & (v72 + v39);
                  v40 = (__int64 *)(v80 + 16LL * v39);
                  v41 = *v40;
                  if ( v8 == *v40 )
                    goto LABEL_53;
                  v72 = v73;
                }
              }
              v42 = *(_QWORD *)(v80 + 16LL * v81 + 8);
              if ( (v42 & 1) == 0 )
              {
LABEL_75:
                v48 = *(_DWORD *)(v42 + 64);
                if ( !v48 )
                  goto LABEL_82;
                v49 = *(_QWORD *)v42;
                _RAX = 0;
                v50 = (unsigned int)(v48 - 1) >> 6;
                v51 = v50;
                v52 = v50 + 1;
                while ( 1 )
                {
                  _RSI = *(_QWORD *)(v49 + 8 * _RAX);
                  v54 = _RAX;
                  if ( v51 == _RAX )
                    break;
                  if ( _RSI )
                    goto LABEL_81;
                  if ( v52 == ++_RAX )
                    goto LABEL_82;
                }
                _RSI &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v48;
                if ( !_RSI )
                  goto LABEL_82;
LABEL_81:
                __asm { tzcnt   rsi, rsi }
                LODWORD(_RAX) = _RSI + ((_DWORD)_RAX << 6);
                if ( (int)_RSI + (v54 << 6) < 0 )
                  goto LABEL_82;
                goto LABEL_56;
              }
LABEL_54:
              if ( !((v42 >> 1) & ~(-1LL << (v42 >> 58))) )
                goto LABEL_82;
              __asm { tzcnt   rax, rax }
LABEL_56:
              v83 = v38;
              v45 = *(_QWORD *)(a1 + 1320) + 2184LL * (int)_RAX;
              if ( *(_DWORD *)(v45 + 32) != 2 )
                goto LABEL_16;
              v77 = *(_QWORD *)(a1 + 8);
              v46 = *(__int64 **)(a1 + 48);
              v82 = *(_QWORD *)(v45 + 40);
              v78 = sub_D95540(v85);
              if ( v78 != sub_D95540(v8)
                || *(_WORD *)(v85 + 24) == 8
                && *(_WORD *)(v8 + 24) == 8
                && *(_QWORD *)(v85 + 48) != *(_QWORD *)(v8 + 48) )
              {
                goto LABEL_16;
              }
              sub_DC06D0((__int64)&v90, v77, v85, v8);
              if ( v92 )
              {
                if ( v91 <= 0x40 )
                {
                  v47 = 0;
                  if ( v91 )
                    v47 = (__int64)(v90 << (64 - (unsigned __int8)v91)) >> (64 - (unsigned __int8)v91);
                }
                else
                {
                  v47 = *(_QWORD *)v90;
                }
                if ( sub_DFA150(v46, v82, 0, v47, 1u, 0) )
                {
                  if ( v91 > 0x40 )
                  {
                    v76 = -*(_QWORD *)v90;
                  }
                  else
                  {
                    v76 = 0;
                    if ( v91 )
                      v76 = -((__int64)(v90 << (64 - (unsigned __int8)v91)) >> (64 - (unsigned __int8)v91));
                  }
                  if ( !sub_DFA150(v46, v82, 0, v76, 1u, 0) )
                  {
                    if ( v92 )
                    {
                      v92 = 0;
                      if ( v91 > 0x40 )
                      {
                        if ( v90 )
                          j_j___libc_free_0_0(v90);
                      }
                    }
                    v83 = v38;
                    goto LABEL_92;
                  }
                }
                if ( v92 )
                {
                  v92 = 0;
                  if ( v91 > 0x40 )
                  {
                    if ( v90 )
                    {
                      j_j___libc_free_0_0(v90);
                      v83 = v38;
                      goto LABEL_16;
                    }
                  }
                }
              }
LABEL_82:
              v83 = v38;
              if ( v88 == ++v7 )
                goto LABEL_17;
            }
            else if ( v88 == ++v7 )
            {
              goto LABEL_17;
            }
          }
          else
          {
            v59 = *(_DWORD *)(a1 + 36304);
            v60 = *(_QWORD *)(a1 + 36288);
            if ( v59 )
            {
              v61 = (v59 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
              v62 = (__int64 *)(v60 + 16LL * v61);
              v63 = *v62;
              if ( v8 == *v62 )
                goto LABEL_90;
              v71 = 1;
              while ( v63 != -4096 )
              {
                v75 = v71 + 1;
                v61 = (v59 - 1) & (v71 + v61);
                v62 = (__int64 *)(v60 + 16LL * v61);
                v63 = *v62;
                if ( v8 == *v62 )
                  goto LABEL_90;
                v71 = v75;
              }
            }
            v62 = (__int64 *)(v60 + 16LL * v59);
LABEL_90:
            v64 = v62[1];
            if ( (v64 & 1) != 0 )
            {
              v83 = sub_39FAC40((v64 >> 1) & ~(-1LL << (v64 >> 58)));
            }
            else if ( *(_QWORD *)v64 == *(_QWORD *)v64 + 8LL * *(unsigned int *)(v64 + 8) )
            {
              v83 = 0;
            }
            else
            {
              v87 = v8;
              v66 = *(__int64 **)v64;
              v67 = *(_QWORD *)v64 + 8LL * *(unsigned int *)(v64 + 8);
              v68 = 0;
              do
              {
                v69 = *v66++;
                v68 += sub_39FAC40(v69);
              }
              while ( v66 != (__int64 *)v67 );
              v83 = v68;
              v8 = v87;
            }
LABEL_92:
            v85 = v8;
            if ( v88 == ++v7 )
              goto LABEL_17;
          }
        }
      }
      v9 = v94;
      v10 = &v94[HIDWORD(v95)];
      if ( v94 != v10 )
      {
        while ( v8 != *v9 )
        {
          if ( v10 == ++v9 )
            goto LABEL_42;
        }
        goto LABEL_16;
      }
      goto LABEL_42;
    }
    v12 = 1;
    v11 = 0;
LABEL_18:
    if ( (v12 & (unsigned __int8)sub_DF9980(*(_QWORD *)(a1 + 48))) != 0 )
      break;
    if ( !v97 )
      goto LABEL_97;
    v17 = v94;
    v14 = HIDWORD(v95);
    v13 = &v94[HIDWORD(v95)];
    if ( v94 == v13 )
    {
LABEL_105:
      if ( HIDWORD(v95) >= (unsigned int)v95 )
      {
LABEL_97:
        sub_C8CC70((__int64)&v93, v11, (__int64)v13, v14, v15, v16);
        goto LABEL_24;
      }
      ++HIDWORD(v95);
      *v13 = v11;
      ++v93;
    }
    else
    {
      while ( v11 != *v17 )
      {
        if ( v13 == ++v17 )
          goto LABEL_105;
      }
    }
LABEL_24:
    v86 = 0;
    v89 = 0;
    v2 = *(_QWORD *)(a1 + 1320);
    v84 = *(unsigned int *)(a1 + 1328);
    if ( *(_DWORD *)(a1 + 1328) )
    {
      while ( 1 )
      {
        v18 = v2 + v86;
        if ( *(_BYTE *)(v2 + v86 + 2148) )
        {
          v19 = *(_QWORD **)(v18 + 2128);
          v20 = &v19[*(unsigned int *)(v18 + 2140)];
          if ( v19 != v20 )
          {
            while ( v11 != *v19 )
            {
              if ( v20 == ++v19 )
                goto LABEL_38;
            }
LABEL_30:
            v21 = *(unsigned int *)(v18 + 768);
            v22 = 0;
            v23 = 0;
            if ( !*(_DWORD *)(v18 + 768) )
              goto LABEL_37;
            do
            {
              while ( 1 )
              {
                v24 = *(_QWORD *)(v18 + 760) + 112 * v23;
                v90 = v11;
                if ( v11 == *(_QWORD *)(v24 + 88) )
                  break;
                v25 = *(_QWORD **)(v24 + 40);
                v26 = &v25[*(unsigned int *)(v24 + 48)];
                if ( v26 != sub_284FCC0(v25, (__int64)v26, (__int64 *)&v90) )
                  break;
                --v21;
                sub_28532A0(v18, v27);
                v22 = 1;
                if ( v21 == v23 )
                  goto LABEL_36;
              }
              ++v23;
            }
            while ( v21 != v23 );
LABEL_36:
            if ( v22 )
            {
              sub_2855860(v18, v89, a1 + 36280);
              v2 = *(_QWORD *)(a1 + 1320);
            }
            else
            {
LABEL_37:
              v2 = *(_QWORD *)(a1 + 1320);
            }
          }
        }
        else
        {
          if ( sub_C8CA60(v18 + 2120, v11) )
            goto LABEL_30;
          v2 = *(_QWORD *)(a1 + 1320);
        }
LABEL_38:
        ++v89;
        v86 += 2184;
        if ( v89 == v84 )
        {
          v3 = *(unsigned int *)(a1 + 1328);
          goto LABEL_2;
        }
      }
    }
    v3 = 0;
  }
LABEL_103:
  if ( !v97 )
    _libc_free((unsigned __int64)v94);
}
