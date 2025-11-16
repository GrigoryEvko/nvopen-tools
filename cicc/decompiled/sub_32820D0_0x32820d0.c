// Function: sub_32820D0
// Address: 0x32820d0
//
__int64 __fastcall sub_32820D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  unsigned int *v8; // r15
  __int64 v9; // rcx
  int v10; // esi
  __int64 v11; // r14
  unsigned __int16 *v12; // rdx
  __int64 v13; // rbx
  unsigned __int16 *v14; // rax
  int v15; // ecx
  __int64 v16; // rax
  bool v17; // al
  __int64 v19; // rax
  int v20; // edi
  __int64 v21; // rax
  unsigned __int16 v22; // cx
  __int64 v23; // r8
  unsigned __int16 v24; // si
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int64 v33; // rbx
  __int64 v34; // rsi
  unsigned __int64 v35; // rbx
  bool v36; // al
  _QWORD *v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // rsi
  __int64 v40; // rdx
  char v41; // r8
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rsi
  unsigned __int64 v45; // rdx
  char v46; // al
  unsigned int v47; // eax
  _WORD *v48; // rdx
  __int64 v49; // rdi
  __int64 v50; // rax
  unsigned __int16 v53; // bx
  unsigned __int16 v54; // ax
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rax
  unsigned __int16 v58; // si
  __int64 v59; // rax
  unsigned __int64 v60; // rcx
  __int64 v61; // rdx
  unsigned __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int16 v67; // [rsp+10h] [rbp-130h]
  unsigned __int64 v68; // [rsp+10h] [rbp-130h]
  unsigned __int16 v69; // [rsp+18h] [rbp-128h]
  unsigned __int16 v70; // [rsp+18h] [rbp-128h]
  unsigned __int64 v71; // [rsp+18h] [rbp-128h]
  __int64 v72; // [rsp+18h] [rbp-128h]
  __int64 v73; // [rsp+20h] [rbp-120h]
  __int64 v74; // [rsp+20h] [rbp-120h]
  char v75; // [rsp+20h] [rbp-120h]
  unsigned __int8 v76; // [rsp+20h] [rbp-120h]
  int v77; // [rsp+20h] [rbp-120h]
  unsigned __int8 v78; // [rsp+20h] [rbp-120h]
  _QWORD *v80; // [rsp+30h] [rbp-110h]
  unsigned __int16 *v82; // [rsp+48h] [rbp-F8h]
  char v83; // [rsp+48h] [rbp-F8h]
  bool v84; // [rsp+48h] [rbp-F8h]
  char v85; // [rsp+48h] [rbp-F8h]
  char v86; // [rsp+48h] [rbp-F8h]
  char v87; // [rsp+48h] [rbp-F8h]
  bool v88; // [rsp+48h] [rbp-F8h]
  bool v89; // [rsp+48h] [rbp-F8h]
  unsigned __int64 v90; // [rsp+48h] [rbp-F8h]
  unsigned int *v92; // [rsp+58h] [rbp-E8h]
  __m128i v93; // [rsp+60h] [rbp-E0h] BYREF
  unsigned __int16 v94; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v95; // [rsp+78h] [rbp-C8h]
  __int16 v96; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v97; // [rsp+88h] [rbp-B8h]
  unsigned __int64 v98; // [rsp+90h] [rbp-B0h]
  __int64 v99; // [rsp+98h] [rbp-A8h]
  __int64 v100; // [rsp+A0h] [rbp-A0h]
  __int64 v101; // [rsp+A8h] [rbp-98h]
  unsigned __int64 v102; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v103; // [rsp+B8h] [rbp-88h]
  unsigned __int64 v104; // [rsp+C0h] [rbp-80h]
  __int64 v105; // [rsp+C8h] [rbp-78h]
  __int64 v106; // [rsp+D0h] [rbp-70h]
  __int64 v107; // [rsp+D8h] [rbp-68h]
  unsigned __int64 v108; // [rsp+E0h] [rbp-60h] BYREF
  __int64 v109; // [rsp+E8h] [rbp-58h]
  __int64 v110; // [rsp+F0h] [rbp-50h]
  __int64 v111; // [rsp+F8h] [rbp-48h]
  __int64 v112; // [rsp+100h] [rbp-40h]
  __int64 v113; // [rsp+108h] [rbp-38h]

  v8 = *(unsigned int **)(a2 + 40);
  v80 = (_QWORD *)a6;
  v92 = &v8[10 * *(unsigned int *)(a2 + 64)];
  if ( v92 != v8 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)v8;
        v12 = *(unsigned __int16 **)(*(_QWORD *)v8 + 48LL);
        v13 = v8[2];
        v14 = &v12[8 * v13];
        v15 = *v14;
        v16 = *((_QWORD *)v14 + 1);
        LOWORD(v108) = v15;
        v109 = v16;
        if ( (_WORD)v15 )
        {
          v9 = (unsigned int)(v15 - 17);
          if ( (unsigned __int16)v9 <= 0xD3u )
            return 0;
        }
        else
        {
          v82 = v12;
          v17 = sub_30070B0((__int64)&v108);
          v12 = v82;
          if ( v17 )
            return 0;
        }
        v10 = *(_DWORD *)(v11 + 24);
        LOBYTE(v9) = v10 == 35 || v10 == 11;
        if ( (_BYTE)v9 )
        {
          if ( (unsigned int)(*(_DWORD *)(a2 + 24) - 187) > 1 )
            goto LABEL_6;
          v30 = *(_QWORD *)(v11 + 96);
          v31 = *(_QWORD *)(a5 + 96);
          v32 = *(unsigned int *)(v31 + 32);
          LODWORD(v103) = v32;
          if ( (unsigned int)v32 <= 0x40 )
          {
            v33 = *(_QWORD *)(v31 + 24);
            goto LABEL_45;
          }
          v72 = v30;
          v76 = v9;
          sub_C43780((__int64)&v102, (const void **)(v31 + 24));
          v32 = (unsigned int)v103;
          v9 = v76;
          v30 = v72;
          if ( (unsigned int)v103 <= 0x40 )
          {
            v33 = v102;
LABEL_45:
            v34 = *(_QWORD *)(v30 + 24);
            LODWORD(v109) = v32;
            LODWORD(v103) = 0;
            v35 = v34 & v33;
            v102 = v35;
            v108 = v35;
          }
          else
          {
            sub_C43B90(&v102, (__int64 *)(v72 + 24));
            v32 = (unsigned int)v103;
            v35 = v102;
            LODWORD(v103) = 0;
            v30 = v72;
            v9 = v76;
            LODWORD(v109) = v32;
            v108 = v102;
            if ( (unsigned int)v32 > 0x40 )
            {
              v36 = !sub_C43C50((__int64)&v108, (const void **)(v72 + 24));
              if ( v35 )
              {
                v88 = v36;
                j_j___libc_free_0_0(v35);
                v36 = v88;
              }
              if ( (unsigned int)v103 > 0x40 )
              {
                if ( v102 )
                {
                  v84 = v36;
                  j_j___libc_free_0_0(v102);
                  v36 = v84;
                }
              }
              if ( !v36 )
                goto LABEL_6;
LABEL_50:
              if ( !*(_BYTE *)(a4 + 28) )
                goto LABEL_88;
              v37 = *(_QWORD **)(a4 + 8);
              v9 = *(unsigned int *)(a4 + 20);
              v32 = (__int64)&v37[v9];
              if ( v37 != (_QWORD *)v32 )
              {
                while ( a2 != *v37 )
                {
                  if ( (_QWORD *)v32 == ++v37 )
                    goto LABEL_54;
                }
                goto LABEL_6;
              }
LABEL_54:
              if ( (unsigned int)v9 < *(_DWORD *)(a4 + 16) )
              {
                *(_DWORD *)(a4 + 20) = v9 + 1;
                *(_QWORD *)v32 = a2;
                ++*(_QWORD *)a4;
              }
              else
              {
LABEL_88:
                sub_C8CC70(a4, a2, v32, v9, v30, a6);
              }
              goto LABEL_6;
            }
            v34 = *(_QWORD *)(v72 + 24);
          }
          if ( v34 == v35 )
            goto LABEL_6;
          goto LABEL_50;
        }
        v19 = *(_QWORD *)(v11 + 56);
        if ( !v19 )
          return 0;
        v20 = 1;
        do
        {
          while ( *(_DWORD *)(v19 + 8) != (_DWORD)v13 )
          {
            v19 = *(_QWORD *)(v19 + 32);
            if ( !v19 )
              goto LABEL_18;
          }
          if ( !v20 )
            return 0;
          v21 = *(_QWORD *)(v19 + 32);
          if ( !v21 )
            goto LABEL_19;
          if ( (_DWORD)v13 == *(_DWORD *)(v21 + 8) )
            return 0;
          v19 = *(_QWORD *)(v21 + 32);
          v20 = 0;
        }
        while ( v19 );
LABEL_18:
        if ( v20 == 1 )
          return 0;
LABEL_19:
        if ( v10 == 214 )
        {
LABEL_70:
          v50 = *(_QWORD *)(a5 + 96);
          if ( *(_DWORD *)(v50 + 32) > 0x40u )
          {
            v77 = *(_DWORD *)(v11 + 24);
            v89 = v10 == 35 || v10 == 11;
            LODWORD(_RAX) = sub_C445E0(v50 + 24);
            v10 = v77;
            LOBYTE(v9) = v89;
            if ( (_DWORD)_RAX == 1 )
              goto LABEL_90;
LABEL_73:
            switch ( (_DWORD)_RAX )
            {
              case 2:
                v53 = 3;
                goto LABEL_91;
              case 4:
                v53 = 4;
                goto LABEL_91;
              case 8:
                v53 = 5;
                goto LABEL_91;
              case 0x10:
                v53 = 6;
                goto LABEL_91;
              case 0x20:
                v53 = 7;
                goto LABEL_91;
            }
            if ( (_DWORD)_RAX != 64 )
            {
              v53 = 9;
              if ( (_DWORD)_RAX == 128 )
                goto LABEL_91;
              v87 = v9;
              v54 = sub_3007020(*(_QWORD **)(*(_QWORD *)a1 + 64LL), _RAX);
              v10 = *(_DWORD *)(v11 + 24);
              LOBYTE(v9) = v87;
              v53 = v54;
              goto LABEL_92;
            }
          }
          else
          {
            _R8 = ~*(_QWORD *)(v50 + 24);
            if ( *(_QWORD *)(v50 + 24) != -1 )
            {
              __asm { tzcnt   rax, r8 }
              if ( (_DWORD)_RAX != 1 )
                goto LABEL_73;
LABEL_90:
              v53 = 2;
LABEL_91:
              v55 = 0;
LABEL_92:
              v94 = v53;
              v56 = *(_QWORD *)(v11 + 40);
              v95 = v55;
              if ( v10 == 4 )
              {
                v64 = *(_QWORD *)(v56 + 40);
                v58 = *(_WORD *)(v64 + 96);
                v59 = *(_QWORD *)(v64 + 104);
              }
              else
              {
                v57 = *(_QWORD *)(*(_QWORD *)v56 + 48LL) + 16LL * *(unsigned int *)(v56 + 8);
                v58 = *(_WORD *)v57;
                v59 = *(_QWORD *)(v57 + 8);
              }
              if ( v58 == v53 )
              {
                if ( v53 || v59 == v55 )
                  goto LABEL_6;
                v109 = v59;
                LOWORD(v108) = 0;
LABEL_57:
                v85 = v9;
                v38 = sub_3007260((__int64)&v108);
                LOBYTE(v9) = v85;
                v112 = v38;
                v39 = v38;
                v113 = v40;
                v41 = v40;
              }
              else
              {
                LOWORD(v108) = v58;
                v109 = v59;
                if ( !v58 )
                  goto LABEL_57;
                if ( v58 == 1 || (unsigned __int16)(v58 - 504) <= 7u )
                  goto LABEL_141;
                v66 = 16LL * (v58 - 1) + 71615648;
                v39 = *(_QWORD *)&byte_444C4A0[16 * v58 - 16];
                v41 = *(_BYTE *)(v66 + 8);
              }
              if ( v53 )
              {
                if ( v53 == 1 || (unsigned __int16)(v53 - 504) <= 7u )
                  goto LABEL_141;
                v45 = *(_QWORD *)&byte_444C4A0[16 * v53 - 16];
                v46 = byte_444C4A0[16 * v53 - 8];
              }
              else
              {
                v71 = v39;
                v75 = v41;
                v86 = v9;
                v42 = sub_3007260((__int64)&v94);
                v41 = v75;
                LOBYTE(v9) = v86;
                v44 = v43;
                v110 = v42;
                v45 = v42;
                v111 = v44;
                v39 = v71;
                v46 = v111;
              }
              if ( !v46 && v41 || v39 > v45 )
              {
LABEL_63:
                if ( *v80 )
                  return 0;
                *v80 = v11;
                v47 = *(_DWORD *)(v11 + 68);
                if ( v47 > 1 )
                {
                  v48 = *(_WORD **)(v11 + 48);
                  v49 = (__int64)&v48[8 * v47];
                  while ( 1 )
                  {
                    if ( *v48 != 262 && *v48 != 1 )
                    {
                      if ( (_BYTE)v9 )
                      {
                        *v80 = 0;
                        return 0;
                      }
                      LOBYTE(v9) = *v48 != 262 && *v48 != 1;
                    }
                    v48 += 8;
                    if ( v48 == (_WORD *)v49 )
                      goto LABEL_6;
                  }
                }
                goto LABEL_6;
              }
              goto LABEL_6;
            }
          }
          v53 = 8;
          goto LABEL_91;
        }
        if ( v10 > 214 )
          break;
        if ( v10 == 4 )
          goto LABEL_70;
        if ( (unsigned int)(v10 - 186) > 2 )
          goto LABEL_63;
        if ( !(unsigned __int8)sub_32820D0(a1, v11, a3, a4, a5, v80) )
          return 0;
        v8 += 10;
        if ( v92 == v8 )
          return 1;
      }
      if ( v10 != 298 )
        goto LABEL_63;
      v93.m128i_i64[1] = 0;
      v93.m128i_i16[0] = 0;
      if ( !(unsigned __int8)sub_3271FA0(a1, *(_QWORD *)(a5 + 96), v11, *v12, &v93)
        || !(unsigned __int8)sub_3281800(a1, v11, 3u, &v93, 0) )
      {
        return 0;
      }
      v22 = *(_WORD *)(v11 + 96);
      v23 = *(_QWORD *)(v11 + 104);
      v24 = v93.m128i_i16[0];
      if ( ((*(_BYTE *)(v11 + 33) ^ 0xC) & 0xC) == 0 )
        break;
LABEL_38:
      if ( v22 == v24 )
      {
        if ( v24 || v23 == v93.m128i_i64[1] )
          goto LABEL_40;
        v103 = v23;
        LOWORD(v102) = 0;
LABEL_100:
        v106 = sub_3007260((__int64)&v102);
        v60 = v106;
        v107 = v61;
        v23 = (unsigned __int8)v61;
        goto LABEL_101;
      }
      LOWORD(v102) = v22;
      v103 = v23;
      if ( !v22 )
        goto LABEL_100;
      if ( v22 == 1 || (unsigned __int16)(v22 - 504) <= 7u )
        goto LABEL_141;
      v65 = 16LL * (v22 - 1) + 71615648;
      v60 = *(_QWORD *)&byte_444C4A0[16 * v22 - 16];
      v23 = *(unsigned __int8 *)(v65 + 8);
LABEL_101:
      if ( v24 )
      {
        if ( v24 == 1 || (unsigned __int16)(v24 - 504) <= 7u )
          goto LABEL_141;
        v62 = *(_QWORD *)&byte_444C4A0[16 * v24 - 16];
        LOBYTE(v63) = byte_444C4A0[16 * v24 - 8];
      }
      else
      {
        v78 = v23;
        v90 = v60;
        v62 = sub_3007260((__int64)&v93);
        v23 = v78;
        v60 = v90;
        v104 = v62;
        v105 = v63;
      }
      if ( (!(_BYTE)v63 || (_BYTE)v23) && v62 <= v60 )
      {
LABEL_40:
        v29 = *(unsigned int *)(a3 + 8);
        if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v29 + 1, 8u, v23, a6);
          v29 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v29) = v11;
        ++*(_DWORD *)(a3 + 8);
      }
LABEL_6:
      v8 += 10;
      if ( v92 == v8 )
        return 1;
    }
    if ( v22 == v93.m128i_i16[0] )
    {
      if ( v22 || v23 == v93.m128i_i64[1] )
        goto LABEL_6;
      v97 = *(_QWORD *)(v11 + 104);
      v96 = 0;
    }
    else
    {
      v96 = *(_WORD *)(v11 + 96);
      v97 = v23;
      if ( v22 )
      {
        if ( v22 == 1 || (unsigned __int16)(v22 - 504) <= 7u )
LABEL_141:
          BUG();
        a6 = *(_QWORD *)&byte_444C4A0[16 * v22 - 16];
        v83 = byte_444C4A0[16 * v22 - 8];
LABEL_33:
        if ( v24 )
        {
          if ( v24 == 1 || (unsigned __int16)(v24 - 504) <= 7u )
            goto LABEL_141;
          v27 = *(_QWORD *)&byte_444C4A0[16 * v24 - 16];
          LOBYTE(v28) = byte_444C4A0[16 * v24 - 8];
        }
        else
        {
          v68 = a6;
          v70 = v22;
          v74 = v23;
          v27 = sub_3007260((__int64)&v93);
          a6 = v68;
          v98 = v27;
          v22 = v70;
          v99 = v28;
          v23 = v74;
        }
        if ( ((_BYTE)v28 || !v83) && v27 >= a6 )
          goto LABEL_6;
        goto LABEL_38;
      }
    }
    v67 = v93.m128i_i16[0];
    v69 = v22;
    v73 = v23;
    v25 = sub_3007260((__int64)&v96);
    v24 = v67;
    v22 = v69;
    v100 = v25;
    a6 = v25;
    v23 = v73;
    v101 = v26;
    v83 = v26;
    goto LABEL_33;
  }
  return 1;
}
