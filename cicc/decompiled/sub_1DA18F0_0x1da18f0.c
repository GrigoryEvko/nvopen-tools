// Function: sub_1DA18F0
// Address: 0x1da18f0
//
unsigned __int64 __fastcall sub_1DA18F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 (*v6)(); // rax
  __int64 v7; // rax
  char *v8; // r8
  __int64 v9; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v11; // r12
  char *v12; // r15
  char v13; // al
  __int16 v14; // ax
  __int64 v15; // rax
  __int64 *v16; // rbx
  __int64 *v17; // r11
  __int64 *v18; // r12
  unsigned int v19; // eax
  __int64 *v20; // rcx
  int v21; // ecx
  int v22; // eax
  int v25; // ecx
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 v28; // r8
  int v29; // edx
  __int64 v30; // rcx
  __int64 v31; // rsi
  int v32; // r11d
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // r9
  unsigned int i; // eax
  _QWORD *v37; // r9
  unsigned int v38; // eax
  unsigned int v39; // eax
  _QWORD *j; // rbx
  _QWORD *v41; // rdi
  int v42; // edx
  __int64 *v43; // rdx
  int v44; // esi
  int v45; // eax
  __int64 v48; // rax
  unsigned __int64 v49; // rax
  __int64 v50; // rax
  int v51; // edx
  char v52; // di
  unsigned int k; // r8d
  __int64 v54; // rdx
  char v55; // r9
  __int64 v56; // rsi
  __int64 v57; // r10
  __int64 v58; // r10
  unsigned __int16 v59; // r11
  __int64 v60; // rcx
  __int64 v61; // rdx
  unsigned __int16 v62; // di
  _WORD *v63; // rsi
  unsigned __int16 *v64; // r9
  unsigned __int16 *v65; // r8
  unsigned __int16 *v66; // rsi
  unsigned __int16 v67; // ax
  __int16 *v68; // rbx
  unsigned __int16 v69; // r15
  __int64 *v70; // rdx
  int v71; // esi
  int v72; // eax
  __int64 v75; // rax
  unsigned __int64 v76; // rax
  __int16 v77; // ax
  unsigned __int16 *v78; // rax
  __int64 v79; // rax
  __int16 v80; // ax
  char *v81; // [rsp+8h] [rbp-D8h]
  _QWORD *v82; // [rsp+20h] [rbp-C0h]
  unsigned __int16 v84; // [rsp+42h] [rbp-9Eh]
  int v85; // [rsp+44h] [rbp-9Ch]
  unsigned __int16 *v86; // [rsp+48h] [rbp-98h]
  __int64 *v87; // [rsp+48h] [rbp-98h]
  unsigned __int16 v88; // [rsp+50h] [rbp-90h]
  __int64 *v89; // [rsp+50h] [rbp-90h]
  _QWORD *v91; // [rsp+60h] [rbp-80h] BYREF
  _QWORD v92[3]; // [rsp+68h] [rbp-78h] BYREF
  char v93[8]; // [rsp+80h] [rbp-60h] BYREF
  _QWORD **v94; // [rsp+88h] [rbp-58h]
  __int64 *v95; // [rsp+90h] [rbp-50h]
  unsigned __int64 v96; // [rsp+98h] [rbp-48h]
  unsigned __int64 v97; // [rsp+A0h] [rbp-40h]

  v6 = *(__int64 (**)())(**(_QWORD **)(sub_1E15F70(a2) + 16) + 56LL);
  if ( v6 == sub_1D12D20 )
    BUG();
  v7 = v6();
  v8 = *(char **)(a2 + 32);
  v92[2] = 0;
  LODWORD(v7) = *(_DWORD *)(v7 + 112);
  v92[1] = v92;
  v92[0] = v92;
  v85 = v7;
  v9 = *(unsigned int *)(a2 + 40);
  v91 = v92;
  result = (unsigned __int64)&v8[40 * v9];
  if ( (char *)result != v8 )
  {
    v11 = result;
    v12 = v8;
    while ( 1 )
    {
      v13 = *v12;
      if ( *v12 )
        goto LABEL_13;
      if ( (v12[3] & 0x10) != 0 && *((int *)v12 + 2) > 0 )
      {
        v14 = *(_WORD *)(a2 + 46);
        if ( (v14 & 4) != 0 || (v14 & 8) == 0 )
          v15 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 4) & 1LL;
        else
          LOBYTE(v15) = sub_1E15D00(a2, 16, 1);
        if ( (_BYTE)v15 && *((_DWORD *)v12 + 2) == v85 )
        {
          v13 = *v12;
LABEL_13:
          if ( v13 == 12 )
          {
            v43 = *(__int64 **)(a3 + 8);
            v94 = (_QWORD **)a3;
            v96 = 0xFFFFFFFF00000000LL;
            v95 = v43;
            v97 = 0;
            v93[0] = 0;
            if ( v43 != (__int64 *)(a3 + 8) )
            {
              v44 = 0;
              v45 = *((_DWORD *)v43 + 4) << 7;
              LODWORD(v96) = v45;
              _RCX = v43[3];
              if ( !_RCX )
              {
                _RCX = v43[4];
                v44 = 64;
                if ( !_RCX )
                {
                  _RCX = v43[5];
                  v44 = 128;
                }
              }
              __asm { tzcnt   rcx, rcx }
              LODWORD(_RCX) = v44 + _RCX;
              LODWORD(v96) = _RCX + v45;
              v48 = ((unsigned int)(_RCX + v45) >> 6) & 1;
              HIDWORD(v96) = v48;
              v97 = (unsigned __int64)v43[v48 + 3] >> _RCX;
              do
              {
                v49 = *(_QWORD *)(a4 + 48) + ((unsigned __int64)(unsigned int)(v96 - 1) << 7);
                if ( *(_DWORD *)(v49 + 112) == 1 )
                {
                  v50 = *(_QWORD *)(v49 + 120);
                  if ( (_DWORD)v50 )
                  {
                    if ( v85 != (_DWORD)v50 )
                    {
                      v51 = *(_DWORD *)(*((_QWORD *)v12 + 3) + 4LL * ((unsigned int)v50 >> 5));
                      if ( !_bittest(&v51, v50) )
                        sub_1369D60((__int64 *)&v91, v96);
                    }
                  }
                }
                sub_1DA06D0((__int64)v93);
              }
              while ( !v93[0] );
            }
          }
          goto LABEL_14;
        }
        v82 = *(_QWORD **)(a1 + 232);
        if ( !v82 )
          BUG();
        v59 = 0;
        v60 = v82[1];
        v61 = v82[7];
        v62 = 0;
        v63 = (_WORD *)(v61 + 2LL * (*(_DWORD *)(v60 + 24LL * *((unsigned int *)v12 + 2) + 16) >> 4));
        v64 = v63 + 1;
        v88 = *v63 + *((_WORD *)v12 + 4) * (*(_WORD *)(v60 + 24LL * *((unsigned int *)v12 + 2) + 16) & 0xF);
LABEL_70:
        v65 = v64;
LABEL_71:
        if ( v65 )
        {
          v66 = (unsigned __int16 *)(v82[6] + 4LL * v88);
          v67 = *v66;
          v59 = v66[1];
          if ( !*v66 )
            goto LABEL_100;
          while ( 1 )
          {
            v68 = (__int16 *)(v61 + 2LL * *(unsigned int *)(v60 + 24LL * v67 + 8));
            if ( v68 )
              break;
            if ( !v59 )
            {
              v62 = v67;
LABEL_100:
              v80 = *v65++;
              v64 = 0;
              if ( !v80 )
                goto LABEL_70;
              v88 += v80;
              goto LABEL_71;
            }
            v67 = v59;
            v59 = 0;
          }
          v86 = v65;
          v84 = v67;
        }
        else
        {
          v86 = 0;
          v68 = 0;
          v84 = v62;
        }
        v81 = v12;
        v69 = v59;
        while ( v86 )
        {
          v70 = *(__int64 **)(a3 + 8);
          v94 = (_QWORD **)a3;
          v96 = 0xFFFFFFFF00000000LL;
          v95 = v70;
          v97 = 0;
          v93[0] = 0;
          if ( v70 != (__int64 *)(a3 + 8) )
          {
            v71 = 0;
            v72 = *((_DWORD *)v70 + 4) << 7;
            LODWORD(v96) = v72;
            _RCX = v70[3];
            if ( !_RCX )
            {
              _RCX = v70[4];
              v71 = 64;
              if ( !_RCX )
              {
                _RCX = v70[5];
                v71 = 128;
              }
            }
            __asm { tzcnt   rcx, rcx }
            LODWORD(_RCX) = v71 + _RCX;
            LODWORD(v96) = _RCX + v72;
            v75 = ((unsigned int)(_RCX + v72) >> 6) & 1;
            HIDWORD(v96) = v75;
            v97 = (unsigned __int64)v70[v75 + 3] >> _RCX;
            do
            {
              v76 = *(_QWORD *)(a4 + 48) + ((unsigned __int64)(unsigned int)(v96 - 1) << 7);
              if ( *(_DWORD *)(v76 + 112) == 1 )
              {
                if ( *(_DWORD *)(v76 + 120) == v84 )
LABEL_91:
                  sub_1369D60((__int64 *)&v91, v96);
              }
              else if ( !v84 )
              {
                goto LABEL_91;
              }
              sub_1DA06D0((__int64)v93);
            }
            while ( !v93[0] );
          }
          v77 = *v68++;
          v84 += v77;
          if ( !v77 )
          {
            if ( v69 )
            {
              v79 = v69;
              v84 = v69;
              v69 = 0;
              v68 = (__int16 *)(v82[7] + 2LL * *(unsigned int *)(v82[1] + 24 * v79 + 8));
            }
            else
            {
              v69 = *v86;
              v88 += *v86;
              if ( *v86 )
              {
                ++v86;
                v78 = (unsigned __int16 *)(v82[6] + 4LL * v88);
                v69 = v78[1];
                v84 = *v78;
                v68 = (__int16 *)(v82[7] + 2LL * *(unsigned int *)(v82[1] + 24LL * *v78 + 8));
              }
              else
              {
                v86 = 0;
                v68 = 0;
              }
            }
          }
        }
        v12 = v81;
      }
LABEL_14:
      v12 += 40;
      if ( (char *)v11 == v12 )
      {
        v16 = (__int64 *)v92[0];
        v17 = (__int64 *)(a3 + 8);
        v18 = *(__int64 **)(a3 + 8);
        if ( v18 == (__int64 *)(a3 + 8) || v92[0] == (_QWORD)v92 )
          goto LABEL_24;
        while ( 1 )
        {
          while ( 1 )
          {
            v19 = *((_DWORD *)v16 + 4);
            if ( *((_DWORD *)v18 + 4) <= v19 )
              break;
            v16 = (__int64 *)*v16;
LABEL_18:
            if ( v16 == v92 || v18 == v17 )
            {
LABEL_23:
              v16 = (__int64 *)v92[0];
              *(_QWORD *)a3 = *(_QWORD *)(a3 + 8);
LABEL_24:
              v95 = v16;
              v94 = &v91;
              result = 0xFFFFFFFF00000000LL;
              v96 = 0xFFFFFFFF00000000LL;
              v97 = 0;
              v93[0] = 0;
              if ( v16 == v92 )
                return result;
              v21 = 0;
              v22 = *((_DWORD *)v16 + 4) << 7;
              LODWORD(v96) = v22;
              _RDX = v16[3];
              if ( !_RDX )
              {
                _RDX = v16[4];
                v21 = 64;
                if ( !_RDX )
                {
                  _RDX = v16[5];
                  v21 = 128;
                }
              }
              __asm { tzcnt   rdx, rdx }
              v25 = _RDX + v21;
              LODWORD(v96) = v25 + v22;
              v26 = ((unsigned int)(v25 + v22) >> 6) & 1;
              HIDWORD(v96) = v26;
              v97 = (unsigned __int64)v16[v26 + 3] >> v25;
              while ( 2 )
              {
                v27 = (__int64 *)(*(_QWORD *)(a4 + 48) + ((unsigned __int64)(unsigned int)(v96 - 1) << 7));
                if ( (*(_BYTE *)(a3 + 40) & 1) != 0 )
                {
                  v28 = a3 + 48;
                  v29 = 7;
LABEL_31:
                  v30 = *v27;
                  v31 = v27[1];
                  v32 = 1;
                  v33 = (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4)
                       | ((unsigned __int64)(((unsigned int)*v27 >> 9) ^ ((unsigned int)*v27 >> 4)) << 32))
                      - 1
                      - ((unsigned __int64)(((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4)) << 32);
                  v34 = ((v33 >> 22) ^ v33) - 1 - (((v33 >> 22) ^ v33) << 13);
                  v35 = ((9 * ((v34 >> 8) ^ v34)) >> 15) ^ (9 * ((v34 >> 8) ^ v34));
                  for ( i = v29 & (((v35 - 1 - (v35 << 27)) >> 31) ^ (v35 - 1 - ((_DWORD)v35 << 27))); ; i = v29 & v38 )
                  {
                    v37 = (_QWORD *)(v28 + 24LL * i);
                    if ( *v37 == v30 && v37[1] == v31 )
                      break;
                    if ( *v37 == -8 && v37[1] == -8 )
                      goto LABEL_37;
                    v38 = v32 + i;
                    ++v32;
                  }
                  *v37 = -16;
                  v37[1] = -16;
                  v39 = *(_DWORD *)(a3 + 40);
                  ++*(_DWORD *)(a3 + 44);
                  *(_DWORD *)(a3 + 40) = (2 * (v39 >> 1) - 2) | v39 & 1;
                }
                else
                {
                  v42 = *(_DWORD *)(a3 + 56);
                  v28 = *(_QWORD *)(a3 + 48);
                  if ( v42 )
                  {
                    v29 = v42 - 1;
                    goto LABEL_31;
                  }
                }
LABEL_37:
                result = sub_1DA06D0((__int64)v93);
                if ( v93[0] )
                {
                  for ( j = (_QWORD *)v92[0]; j != v92; result = j_j___libc_free_0(v41, 40) )
                  {
                    v41 = j;
                    j = (_QWORD *)*j;
                  }
                  return result;
                }
                continue;
              }
            }
          }
          v20 = (__int64 *)*v18;
          if ( *((_DWORD *)v18 + 4) == v19 )
          {
            v52 = 1;
            for ( k = 0; ; k = 1 )
            {
              v54 = k;
              v55 = 0;
              v56 = v18[k + 3];
              v57 = v56;
              while ( 1 )
              {
                while ( 1 )
                {
                  v58 = ~v16[v54 + 3] & v57;
                  v18[v54 + 3] = v58;
                  if ( v58 )
                    v52 = 0;
                  if ( !v55 )
                    break;
                  if ( k == 1 )
                    goto LABEL_64;
                  v57 = v18[4];
                  k = 1;
                  v56 = 0;
                  v54 = 1;
                }
                if ( v58 == v56 )
                  break;
                if ( k == 1 )
                  goto LABEL_64;
                v57 = v18[4];
                v55 = 1;
                k = 1;
                v56 = 0;
                v54 = 1;
              }
              if ( k == 1 )
                break;
            }
LABEL_64:
            if ( v52 )
            {
              --*(_QWORD *)(a3 + 24);
              v87 = v17;
              v89 = v20;
              sub_2208CA0(v18);
              j_j___libc_free_0(v18, 40);
              v17 = v87;
              v20 = v89;
            }
            v16 = (__int64 *)*v16;
            v18 = v20;
            goto LABEL_18;
          }
          v18 = (__int64 *)*v18;
          if ( v20 == v17 )
            goto LABEL_23;
        }
      }
    }
  }
  return result;
}
