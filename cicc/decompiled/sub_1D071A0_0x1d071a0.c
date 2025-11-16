// Function: sub_1D071A0
// Address: 0x1d071a0
//
void __fastcall sub_1D071A0(__int64 **a1)
{
  __int64 *v1; // rax
  unsigned int *v2; // r15
  __int64 *v3; // r14
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 i; // r13
  unsigned int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 *v10; // r13
  int v11; // eax
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v16; // r8
  int v17; // edx
  unsigned int v18; // eax
  int v19; // r14d
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rdx
  __int64 v24; // rdx
  int v25; // r9d
  __int64 v26; // rax
  unsigned __int64 v27; // r14
  unsigned __int16 *v28; // r14
  unsigned __int16 v29; // ax
  __int64 *v30; // rbx
  unsigned __int16 *v31; // r13
  __int64 v32; // rcx
  int v33; // eax
  unsigned int *v34; // rax
  int v35; // ebx
  __int64 v36; // rcx
  int v37; // r8d
  int v38; // r9d
  __int64 *v39; // rbx
  unsigned __int64 v40; // rax
  __int64 v41; // r10
  unsigned int v42; // esi
  __int64 v43; // rdx
  __int64 *v44; // r9
  __int64 v45; // r8
  int v46; // r11d
  __int64 v47; // rax
  __int64 *v48; // r13
  __int64 *v49; // rcx
  __int64 v50; // rdi
  __int64 *v51; // rbx
  __int64 v52; // rdi
  int j; // eax
  unsigned int *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rcx
  int v57; // r8d
  int v58; // r8d
  int v59; // r9d
  __int64 v60; // rax
  unsigned int *v61; // r8
  __int64 v62; // r15
  __int64 v63; // r12
  __int64 v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rdi
  unsigned int v67; // ebx
  __int64 v68; // rdx
  _QWORD *v69; // rax
  unsigned int v70; // r14d
  int v71; // edx
  unsigned int *v72; // rax
  __int64 v73; // r15
  unsigned int v74; // r12d
  unsigned int *v75; // rbx
  signed int v76; // esi
  unsigned int *v77; // rax
  int v78; // r8d
  int v79; // r13d
  int v80; // r13d
  __int64 v81; // r10
  unsigned int v82; // esi
  int v83; // eax
  __int64 *v84; // rdi
  int v85; // eax
  __int64 *v86; // rbx
  __int64 *v87; // r12
  __int64 v88; // rax
  int v89; // r13d
  int v90; // r13d
  __int64 v91; // r10
  unsigned int v92; // esi
  int v93; // r11d
  int v94; // r11d
  __int64 v95; // [rsp+18h] [rbp-D8h]
  __int64 *v96; // [rsp+20h] [rbp-D0h]
  __int64 v98; // [rsp+38h] [rbp-B8h]
  unsigned int *v99; // [rsp+38h] [rbp-B8h]
  __int64 v100; // [rsp+38h] [rbp-B8h]
  unsigned int v101; // [rsp+4Ch] [rbp-A4h] BYREF
  _BYTE *v102; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v103; // [rsp+58h] [rbp-98h]
  _BYTE v104[16]; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v105; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int64 v106; // [rsp+78h] [rbp-78h] BYREF
  __int64 v107; // [rsp+80h] [rbp-70h] BYREF
  _BYTE v108[16]; // [rsp+88h] [rbp-68h] BYREF
  int v109; // [rsp+98h] [rbp-58h] BYREF
  __int64 v110; // [rsp+A0h] [rbp-50h]
  int *v111; // [rsp+A8h] [rbp-48h]
  int *v112; // [rsp+B0h] [rbp-40h]
  __int64 v113; // [rsp+B8h] [rbp-38h]

  v1 = *a1;
  if ( !**a1 )
    return;
  v2 = (unsigned int *)&v102;
  while ( 1 )
  {
    v103 = 0x400000000LL;
    v102 = v104;
    v3 = a1[1];
    if ( !*((_DWORD *)v3 + 181) )
      return;
    v4 = *v1;
    v106 = 0x400000000LL;
    v105 = &v107;
    v109 = 0;
    v110 = 0;
    v111 = &v109;
    v112 = &v109;
    v113 = 0;
    v5 = *(_QWORD *)(v4 + 32);
    for ( i = v5 + 16LL * *(unsigned int *)(v4 + 40); i != v5; v5 += 16 )
    {
      if ( (*(_QWORD *)v5 & 6) == 0 )
      {
        v7 = *(_DWORD *)(v5 + 8);
        if ( v7 )
        {
          v8 = v3[91];
          if ( v4 != *(_QWORD *)(v8 + 8LL * v7) )
            sub_1D04590(*(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL, v7, v8, (__int64)&v105, v2, (_QWORD *)v3[3]);
        }
      }
    }
    v9 = *(_QWORD *)v4;
    v10 = v3;
    while ( v9 )
    {
      v11 = *(__int16 *)(v9 + 24);
      if ( *(_WORD *)(v9 + 24) != 193 )
      {
        if ( (v11 & 0x8000u) != 0 )
        {
          v12 = v10[2];
          if ( ~v11 == *(_DWORD *)(v12 + 40) )
          {
            v101 = *(_DWORD *)(v10[3] + 16);
            if ( *(_QWORD *)(v10[91] + 8LL * v101) )
            {
              v52 = **(_QWORD **)(v10[92] + 8LL * v101);
              for ( j = *(_DWORD *)(v52 + 56); j; j = *(_DWORD *)(*(_QWORD *)v54 + 56LL) )
              {
                v54 = (unsigned int *)(*(_QWORD *)(v52 + 32) + 40LL * (unsigned int)(j - 1));
                if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v54 + 40LL) + 16LL * v54[2]) != 111 )
                  break;
                v52 = *(_QWORD *)v54;
              }
              if ( !(unsigned __int8)sub_1D00BE0(v52, v9, 0, v12)
                && (unsigned __int8)((unsigned __int64)sub_1D041C0((__int64)&v105, &v101, v55, v56, v57) >> 32) )
              {
                v60 = (unsigned int)v103;
                if ( (unsigned int)v103 >= HIDWORD(v103) )
                {
                  sub_16CD150((__int64)v2, v104, 0, 4, v58, v59);
                  v60 = (unsigned int)v103;
                }
                *(_DWORD *)&v102[4 * v60] = v101;
                LODWORD(v103) = v103 + 1;
              }
            }
          }
          v13 = *(_QWORD *)(v9 + 32);
          v14 = v13 + 40LL * *(unsigned int *)(v9 + 56);
          if ( v13 != v14 )
          {
            while ( *(_WORD *)(*(_QWORD *)v13 + 24LL) != 9 )
            {
              v13 += 40;
              if ( v14 == v13 )
                goto LABEL_29;
            }
            v15 = *(_QWORD *)(*(_QWORD *)v13 + 88LL);
            if ( v15 )
            {
              v16 = v10[91];
              v17 = *(_DWORD *)(v10[3] + 16);
              v18 = 1;
              v101 = 1;
              if ( v17 != 2 )
              {
                v96 = v10;
                v20 = v15;
                v95 = v9;
                v21 = v4;
                v22 = v16;
                v19 = v17 - 1;
                do
                {
                  v23 = *(_QWORD *)(v22 + 8LL * v18);
                  if ( v23 )
                  {
                    if ( v21 != v23 )
                    {
                      v24 = *(unsigned int *)(v20 + 4LL * (v18 >> 5));
                      if ( !_bittest((const int *)&v24, v18) )
                      {
                        if ( (unsigned __int8)((unsigned __int64)sub_1D041C0((__int64)&v105, &v101, v24, v14, v16) >> 32) )
                        {
                          v26 = (unsigned int)v103;
                          if ( (unsigned int)v103 >= HIDWORD(v103) )
                          {
                            sub_16CD150((__int64)v2, v104, 0, 4, v16, v25);
                            v26 = (unsigned int)v103;
                          }
                          v14 = v101;
                          *(_DWORD *)&v102[4 * v26] = v101;
                          LODWORD(v103) = v103 + 1;
                        }
                      }
                    }
                  }
                  v18 = v101 + 1;
                  v101 = v18;
                }
                while ( v19 != v18 );
                v4 = v21;
                v10 = v96;
                v9 = v95;
              }
            }
          }
LABEL_29:
          v27 = *(_QWORD *)(v10[2] + 8) + ((unsigned __int64)(unsigned int)~*(__int16 *)(v9 + 24) << 6);
          if ( (*(_BYTE *)(v27 + 8) & 2) != 0 && *(_BYTE *)(v27 + 4) )
          {
            v61 = v2;
            v62 = v4;
            v63 = v9;
            v64 = 0;
            do
            {
              while ( (*(_BYTE *)(*(_QWORD *)(v27 + 40) + 8 * v64 + 2) & 4) == 0 )
              {
                if ( *(unsigned __int8 *)(v27 + 4) <= (unsigned int)++v64 )
                  goto LABEL_67;
              }
              v65 = 5LL * (unsigned int)(v64++ - *(_DWORD *)(v63 + 60));
              v99 = v61;
              sub_1D04590(
                v62,
                *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v63 + 32) + 8 * v65) + 84LL),
                v10[91],
                (__int64)&v105,
                v61,
                (_QWORD *)v10[3]);
              v61 = v99;
            }
            while ( *(unsigned __int8 *)(v27 + 4) > (unsigned int)v64 );
LABEL_67:
            v9 = v63;
            v4 = v62;
            v2 = v61;
          }
          v28 = *(unsigned __int16 **)(v27 + 32);
          if ( v28 )
          {
            v29 = *v28;
            if ( *v28 )
            {
              v98 = v9;
              v30 = v10;
              v31 = v28;
              do
              {
                ++v31;
                sub_1D04590(v4, v29, v30[91], (__int64)&v105, v2, (_QWORD *)v30[3]);
                v29 = *v31;
              }
              while ( *v31 );
              v10 = v30;
              v9 = v98;
            }
          }
        }
        v32 = *(_QWORD *)(v9 + 32);
        v33 = *(_DWORD *)(v9 + 56);
        goto LABEL_36;
      }
      v33 = *(_DWORD *)(v9 + 56);
      v32 = *(_QWORD *)(v9 + 32);
      v66 = (unsigned int)(v33 - 1);
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v32 + 40 * v66) + 40LL) + 16LL * *(unsigned int *)(v32 + 40 * v66 + 8)) != 111 )
        LODWORD(v66) = *(_DWORD *)(v9 + 56);
      if ( (_DWORD)v66 != 4 )
      {
        v100 = v9;
        v67 = 4;
        while ( 1 )
        {
          v68 = *(_QWORD *)(*(_QWORD *)(v32 + 40LL * v67) + 88LL);
          v69 = *(_QWORD **)(v68 + 24);
          if ( *(_DWORD *)(v68 + 32) > 0x40u )
            v69 = (_QWORD *)*v69;
          v70 = v67 + 1;
          v71 = (unsigned __int16)v69 >> 3;
          v67 += v71 + 1;
          if ( ((unsigned __int8)v69 & 7u) - 2 > 2 )
            goto LABEL_72;
          if ( (unsigned __int16)v69 >> 3 )
          {
            v72 = v2;
            v73 = v4;
            v74 = v71 + v70;
            v75 = v72;
            do
            {
              v76 = *(_DWORD *)(*(_QWORD *)(v32 + 40LL * v70) + 84LL);
              if ( v76 > 0 )
              {
                sub_1D04590(v73, v76, v10[91], (__int64)&v105, v75, (_QWORD *)v10[3]);
                v32 = *(_QWORD *)(v100 + 32);
              }
              ++v70;
            }
            while ( v70 != v74 );
            v77 = v75;
            v67 = v74;
            v4 = v73;
            v2 = v77;
            if ( (_DWORD)v66 == v67 )
            {
LABEL_82:
              v33 = *(_DWORD *)(v100 + 56);
              break;
            }
          }
          else
          {
            v67 = v70;
LABEL_72:
            if ( (_DWORD)v66 == v67 )
              goto LABEL_82;
          }
        }
      }
LABEL_36:
      if ( v33 )
      {
        v34 = (unsigned int *)(v32 + 40LL * (unsigned int)(v33 - 1));
        v9 = *(_QWORD *)v34;
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v34 + 40LL) + 16LL * v34[2]) == 111 )
          continue;
      }
      break;
    }
    v35 = v103;
    sub_1D02460(v110);
    if ( v105 != &v107 )
      _libc_free((unsigned __int64)v105);
    if ( !v35 )
      break;
    v39 = a1[1];
    v40 = **a1;
    v41 = (__int64)(v39 + 99);
    v106 = (unsigned __int64)v108;
    v105 = (__int64 *)v40;
    v107 = 0x400000000LL;
    if ( (_DWORD)v103 )
    {
      sub_1D012E0((__int64)&v106, (__int64)v2, (unsigned int)v103, v36, v37, v38);
      v42 = *((_DWORD *)v39 + 204);
      v41 = (__int64)(v39 + 99);
      if ( !v42 )
      {
LABEL_84:
        ++v39[99];
        goto LABEL_85;
      }
    }
    else
    {
      v42 = *((_DWORD *)v39 + 204);
      if ( !v42 )
        goto LABEL_84;
    }
    v43 = (__int64)v105;
    LODWORD(v44) = v42 - 1;
    v45 = v39[100];
    v46 = 1;
    LODWORD(v47) = (v42 - 1) & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
    v48 = (__int64 *)(v45 + 40LL * (unsigned int)v47);
    v49 = 0;
    v50 = *v48;
    if ( v105 != (__int64 *)*v48 )
    {
      while ( v50 != -8 )
      {
        if ( !v49 && v50 == -16 )
          v49 = v48;
        v47 = (unsigned int)v44 & ((_DWORD)v47 + v46);
        v48 = (__int64 *)(v45 + 40 * v47);
        v50 = *v48;
        if ( v105 == (__int64 *)*v48 )
          goto LABEL_44;
        ++v46;
      }
      v85 = *((_DWORD *)v39 + 202);
      if ( !v49 )
        v49 = v48;
      ++v39[99];
      v83 = v85 + 1;
      if ( 4 * v83 >= 3 * v42 )
      {
LABEL_85:
        sub_1CFDB40(v41, 2 * v42);
        v79 = *((_DWORD *)v39 + 204);
        if ( !v79 )
          goto LABEL_127;
        v43 = (__int64)v105;
        v80 = v79 - 1;
        v81 = v39[100];
        v82 = v80 & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
        v49 = (__int64 *)(v81 + 40LL * v82);
        v44 = (__int64 *)*v49;
        v83 = *((_DWORD *)v39 + 202) + 1;
        if ( (__int64 *)*v49 != v105 )
        {
          v78 = 1;
          v84 = 0;
          while ( v44 != (__int64 *)-8LL )
          {
            if ( v44 == (__int64 *)-16LL && !v84 )
              v84 = v49;
            v94 = v78 + 1;
            v78 += v82;
            v82 = v80 & v78;
            v49 = (__int64 *)(v81 + 40LL * (v80 & (unsigned int)v78));
            v44 = (__int64 *)*v49;
            if ( v105 == (__int64 *)*v49 )
              goto LABEL_103;
            v78 = v94;
          }
LABEL_89:
          if ( v84 )
            v49 = v84;
        }
      }
      else
      {
        v78 = v42 >> 3;
        if ( v42 - *((_DWORD *)v39 + 203) - v83 <= v42 >> 3 )
        {
          sub_1CFDB40(v41, v42);
          v89 = *((_DWORD *)v39 + 204);
          if ( !v89 )
          {
LABEL_127:
            ++*((_DWORD *)v39 + 202);
            BUG();
          }
          v43 = (__int64)v105;
          v90 = v89 - 1;
          v78 = 1;
          v84 = 0;
          v91 = v39[100];
          v92 = v90 & (((unsigned int)v105 >> 9) ^ ((unsigned int)v105 >> 4));
          v49 = (__int64 *)(v91 + 40LL * v92);
          v44 = (__int64 *)*v49;
          v83 = *((_DWORD *)v39 + 202) + 1;
          if ( (__int64 *)*v49 != v105 )
          {
            while ( v44 != (__int64 *)-8LL )
            {
              if ( !v84 && v44 == (__int64 *)-16LL )
                v84 = v49;
              v93 = v78 + 1;
              v78 += v92;
              v92 = v90 & v78;
              v49 = (__int64 *)(v91 + 40LL * (v90 & (unsigned int)v78));
              v44 = (__int64 *)*v49;
              if ( v105 == (__int64 *)*v49 )
                goto LABEL_103;
              v78 = v93;
            }
            goto LABEL_89;
          }
        }
      }
LABEL_103:
      *((_DWORD *)v39 + 202) = v83;
      if ( *v49 != -8 )
        --*((_DWORD *)v39 + 203);
      *v49 = v43;
      v49[1] = (__int64)(v49 + 3);
      v49[2] = 0x400000000LL;
      if ( (_DWORD)v107 )
        sub_1D011A0((__int64)(v49 + 1), (char **)&v106, v43, (__int64)v49, v78, (int)v44);
      if ( (_BYTE *)v106 != v108 )
        _libc_free(v106);
      *(_BYTE *)(**a1 + 229) |= 1u;
      v86 = a1[1];
      v87 = *a1;
      v88 = *((unsigned int *)v86 + 188);
      if ( (unsigned int)v88 >= *((_DWORD *)v86 + 189) )
      {
        sub_16CD150((__int64)(v86 + 93), v86 + 95, 0, 8, v78, (int)v44);
        v88 = *((unsigned int *)v86 + 188);
      }
      *(_QWORD *)(v86[93] + 8 * v88) = *v87;
      ++*((_DWORD *)v86 + 188);
      goto LABEL_47;
    }
LABEL_44:
    if ( (_BYTE *)v106 != v108 )
      _libc_free(v106);
    sub_1D012E0((__int64)(v48 + 1), (__int64)v2, v43, (__int64)v49, v45, (int)v44);
LABEL_47:
    v51 = *a1;
    *v51 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1[1][84] + 96LL))(a1[1][84]);
    if ( v102 != v104 )
      _libc_free((unsigned __int64)v102);
    v1 = *a1;
    if ( !**a1 )
      return;
  }
  if ( v102 != v104 )
    _libc_free((unsigned __int64)v102);
}
