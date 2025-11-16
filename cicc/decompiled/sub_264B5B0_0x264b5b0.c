// Function: sub_264B5B0
// Address: 0x264b5b0
//
__int64 __fastcall sub_264B5B0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 *a5,
        _QWORD *a6,
        __m128i a7)
{
  unsigned __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // r9
  unsigned __int64 v10; // r12
  unsigned int v11; // eax
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rcx
  unsigned __int64 *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 j; // rbx
  __int64 v19; // r15
  const char *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r9
  unsigned __int8 *v24; // rax
  unsigned __int8 *v25; // r12
  __int64 v26; // rax
  unsigned __int64 *v27; // rbx
  __int64 v28; // r8
  unsigned __int64 *v29; // r13
  unsigned __int64 v30; // rdi
  _QWORD *v31; // rdx
  _QWORD *v32; // rdi
  _QWORD *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rcx
  __int64 v37; // r15
  __int64 v38; // rcx
  __int64 v39; // rax
  __int64 v40; // rsi
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  char v43; // di
  __int64 *v44; // rax
  __int64 v45; // rdx
  __int64 *v46; // r13
  __int64 v47; // r15
  __int64 *v48; // rbx
  const char *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r8
  __int64 v52; // r9
  unsigned __int8 *v53; // rax
  unsigned __int8 *v54; // r14
  __int64 v55; // rax
  unsigned __int8 *v56; // r12
  __int64 *v57; // rax
  unsigned int v58; // eax
  _QWORD *v59; // r13
  unsigned __int64 v60; // r15
  __int64 v61; // rax
  _QWORD *v62; // r15
  __int64 v63; // rdx
  __int64 v64; // rax
  unsigned int v65; // eax
  _QWORD *v66; // rbx
  _QWORD *v67; // r13
  __int64 v68; // rsi
  unsigned __int64 v69; // rdi
  _QWORD *v70; // rbx
  unsigned __int64 *v71; // rax
  int v72; // r13d
  _QWORD *v73; // rbx
  unsigned __int64 v74; // rdi
  int v75; // r12d
  __int64 v76; // [rsp+10h] [rbp-2B0h]
  _QWORD *v79; // [rsp+28h] [rbp-298h]
  unsigned __int8 *v83; // [rsp+58h] [rbp-268h]
  __int64 i; // [rsp+60h] [rbp-260h]
  __int32 v86; // [rsp+6Ch] [rbp-254h]
  __int64 v87[2]; // [rsp+70h] [rbp-250h] BYREF
  __int64 v88; // [rsp+80h] [rbp-240h] BYREF
  __int64 *v89; // [rsp+90h] [rbp-230h] BYREF
  unsigned __int64 v90; // [rsp+98h] [rbp-228h] BYREF
  __int64 v91; // [rsp+A0h] [rbp-220h] BYREF
  __int64 v92; // [rsp+A8h] [rbp-218h]
  __int64 *v93; // [rsp+B0h] [rbp-210h]
  __int64 v94; // [rsp+C0h] [rbp-200h] BYREF
  const char *v95; // [rsp+E0h] [rbp-1E0h] BYREF
  _QWORD v96[2]; // [rsp+E8h] [rbp-1D8h] BYREF
  __int64 v97; // [rsp+F8h] [rbp-1C8h]
  __int64 v98; // [rsp+100h] [rbp-1C0h]
  unsigned __int64 *v99; // [rsp+130h] [rbp-190h]
  unsigned int v100; // [rsp+138h] [rbp-188h]
  char v101; // [rsp+140h] [rbp-180h] BYREF

  v7 = a3 - 1;
  v76 = a1 + 16;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  if ( v7 > 4 )
  {
    v73 = (_QWORD *)sub_C8D7D0(a1, v76, v7, 8u, (unsigned __int64 *)&v95, (__int64)a6);
    sub_264AB90(a1, v73);
    v74 = *(_QWORD *)a1;
    v75 = (int)v95;
    if ( v76 != v74 )
      _libc_free(v74);
    *(_QWORD *)a1 = v73;
    *(_DWORD *)(a1 + 12) = v75;
  }
  v86 = 1;
  v79 = a6 + 1;
  if ( a3 > 1 )
  {
    do
    {
      v8 = (_QWORD *)sub_22077B0(0x50u);
      v10 = (unsigned __int64)v8;
      if ( v8 )
      {
        *v8 = 0;
        v11 = sub_AF1560(0x56u);
        *(_DWORD *)(v10 + 24) = v11;
        if ( v11 )
        {
          *(_QWORD *)(v10 + 8) = sub_C7D670((unsigned __int64)v11 << 6, 8);
          sub_23FE7B0(v10);
        }
        else
        {
          *(_QWORD *)(v10 + 8) = 0;
          *(_QWORD *)(v10 + 16) = 0;
        }
        *(_BYTE *)(v10 + 64) = 0;
      }
      v12 = *(unsigned int *)(a1 + 8);
      v13 = v12;
      if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v12 )
      {
        v70 = (_QWORD *)sub_C8D7D0(a1, v76, 0, 8u, (unsigned __int64 *)&v95, v9);
        v71 = &v70[*(unsigned int *)(a1 + 8)];
        if ( v71 )
        {
          *v71 = v10;
          v10 = 0;
        }
        sub_264AB90(a1, v70);
        v72 = (int)v95;
        if ( v76 != *(_QWORD *)a1 )
          _libc_free(*(_QWORD *)a1);
        ++*(_DWORD *)(a1 + 8);
        *(_QWORD *)a1 = v70;
        *(_DWORD *)(a1 + 12) = v72;
      }
      else
      {
        v14 = *(_QWORD *)a1;
        v15 = (unsigned __int64 *)(*(_QWORD *)a1 + 8 * v12);
        if ( v15 )
        {
          *v15 = v10;
          ++*(_DWORD *)(a1 + 8);
          goto LABEL_10;
        }
        v14 = a1;
        *(_DWORD *)(a1 + 8) = v13 + 1;
      }
      if ( v10 )
      {
        if ( *(_BYTE *)(v10 + 64) )
        {
          v65 = *(_DWORD *)(v10 + 56);
          *(_BYTE *)(v10 + 64) = 0;
          if ( v65 )
          {
            v66 = *(_QWORD **)(v10 + 40);
            v67 = &v66[2 * v65];
            do
            {
              if ( *v66 != -8192 && *v66 != -4096 )
              {
                v68 = v66[1];
                if ( v68 )
                  sub_B91220((__int64)(v66 + 1), v68);
              }
              v66 += 2;
            }
            while ( v67 != v66 );
            v65 = *(_DWORD *)(v10 + 56);
          }
          sub_C7D6A0(*(_QWORD *)(v10 + 40), 16LL * v65, 8);
        }
        v58 = *(_DWORD *)(v10 + 24);
        if ( v58 )
        {
          v59 = *(_QWORD **)(v10 + 8);
          v90 = 2;
          v60 = (unsigned __int64)v58 << 6;
          v91 = 0;
          v61 = -4096;
          v62 = (_QWORD *)((char *)v59 + v60);
          v92 = -4096;
          v89 = (__int64 *)&unk_49DD7B0;
          v93 = 0;
          v96[0] = 2;
          v96[1] = 0;
          v97 = -8192;
          v95 = (const char *)&unk_49DD7B0;
          v98 = 0;
          while ( 1 )
          {
            v63 = v59[3];
            if ( v61 != v63 )
            {
              v61 = v97;
              if ( v63 != v97 )
              {
                v64 = v59[7];
                if ( v64 != -4096 && v64 != 0 && v64 != -8192 )
                {
                  sub_BD60C0(v59 + 5);
                  v63 = v59[3];
                }
                v61 = v63;
              }
            }
            *v59 = &unk_49DB368;
            if ( v61 != 0 && v61 != -4096 && v61 != -8192 )
              sub_BD60C0(v59 + 1);
            v59 += 8;
            if ( v62 == v59 )
              break;
            v61 = v92;
          }
          v95 = (const char *)&unk_49DB368;
          sub_D68D70(v96);
          v89 = (__int64 *)&unk_49DB368;
          sub_D68D70(&v90);
          v58 = *(_DWORD *)(v10 + 24);
        }
        sub_C7D6A0(*(_QWORD *)(v10 + 8), (unsigned __int64)v58 << 6, 8);
        j_j___libc_free_0(v10);
      }
LABEL_10:
      v16 = sub_F4BFF0(a2, *(_QWORD *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8) - 8), 0, v14);
      v17 = *(_QWORD *)(v16 + 80);
      v83 = (unsigned __int8 *)v16;
      for ( i = v16 + 72; i != v17; v17 = *(_QWORD *)(v17 + 8) )
      {
        if ( !v17 )
          BUG();
        for ( j = *(_QWORD *)(v17 + 32); v17 + 24 != j; j = *(_QWORD *)(j + 8) )
        {
          v19 = j - 24;
          if ( !j )
            v19 = 0;
          sub_B99FD0(v19, 0x22u, 0);
          sub_B99FD0(v19, 0x23u, 0);
        }
      }
      v20 = sub_BD5D20(a2);
      LOWORD(v98) = 261;
      v96[0] = v21;
      v95 = v20;
      sub_2644DA0(v87, v86, v21, v22, 261, v23, a7);
      v24 = sub_BA8CB0(a4, v87[0], v87[1]);
      v25 = v24;
      if ( v24 )
      {
        sub_BD6B90(v83, v24);
        sub_BD84D0((__int64)v25, (__int64)v83);
        sub_B2E860(v25);
      }
      else
      {
        v95 = (const char *)v87;
        LOWORD(v98) = 260;
        sub_BD6B50(v83, &v95);
      }
      sub_B17560((__int64)&v95, (__int64)"memprof-context-disambiguation", (__int64)"MemprofClone", 12, a2);
      sub_B18290((__int64)&v95, "created clone ", 0xEu);
      sub_B16080((__int64)&v89, "NewFunction", 11, v83);
      v26 = sub_23FD640((__int64)&v95, (__int64)&v89);
      sub_1049740(a5, v26);
      if ( v93 != &v94 )
        j_j___libc_free_0((unsigned __int64)v93);
      if ( v89 != &v91 )
        j_j___libc_free_0((unsigned __int64)v89);
      v27 = v99;
      v95 = (const char *)&unk_49D9D40;
      v28 = 10LL * v100;
      v29 = &v99[v28];
      if ( v99 != &v99[v28] )
      {
        do
        {
          v29 -= 10;
          v30 = v29[4];
          if ( (unsigned __int64 *)v30 != v29 + 6 )
            j_j___libc_free_0(v30);
          if ( (unsigned __int64 *)*v29 != v29 + 2 )
            j_j___libc_free_0(*v29);
        }
        while ( v27 != v29 );
        v29 = v99;
      }
      if ( v29 != (unsigned __int64 *)&v101 )
        _libc_free((unsigned __int64)v29);
      v31 = (_QWORD *)a6[2];
      if ( !v31 )
        goto LABEL_39;
      v32 = a6 + 1;
      v33 = (_QWORD *)a6[2];
      do
      {
        while ( 1 )
        {
          v34 = v33[2];
          v35 = v33[3];
          if ( v33[4] >= a2 )
            break;
          v33 = (_QWORD *)v33[3];
          if ( !v35 )
            goto LABEL_37;
        }
        v32 = v33;
        v33 = (_QWORD *)v33[2];
      }
      while ( v34 );
LABEL_37:
      if ( v79 != v32 && v32[4] <= a2 )
      {
        v37 = (__int64)(a6 + 1);
        do
        {
          while ( 1 )
          {
            v38 = v31[2];
            v39 = v31[3];
            if ( v31[4] >= a2 )
              break;
            v31 = (_QWORD *)v31[3];
            if ( !v39 )
              goto LABEL_47;
          }
          v37 = (__int64)v31;
          v31 = (_QWORD *)v31[2];
        }
        while ( v38 );
LABEL_47:
        if ( v79 == (_QWORD *)v37 || *(_QWORD *)(v37 + 32) > a2 )
        {
          v40 = v37;
          v37 = sub_22077B0(0x50u);
          *(_QWORD *)(v37 + 40) = 0;
          *(_QWORD *)(v37 + 32) = a2;
          *(_QWORD *)(v37 + 48) = v37 + 72;
          *(_QWORD *)(v37 + 56) = 1;
          *(_DWORD *)(v37 + 64) = 0;
          *(_BYTE *)(v37 + 68) = 1;
          v41 = sub_264B4B0(a6, v40, (unsigned __int64 *)(v37 + 32));
          if ( v42 )
          {
            v43 = v79 == v42 || v41 || v42[4] > a2;
            sub_220F040(v43, v37, v42, v79);
            ++a6[5];
          }
          else
          {
            v69 = v37;
            v37 = (__int64)v41;
            j_j___libc_free_0(v69);
          }
        }
        v44 = *(__int64 **)(v37 + 48);
        if ( *(_BYTE *)(v37 + 68) )
          v45 = *(unsigned int *)(v37 + 60);
        else
          v45 = *(unsigned int *)(v37 + 56);
        v46 = &v44[v45];
        if ( v44 != v46 )
        {
          while ( 1 )
          {
            v47 = *v44;
            v48 = v44;
            if ( (unsigned __int64)*v44 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v46 == ++v44 )
              goto LABEL_59;
          }
          if ( v44 != v46 )
          {
            do
            {
              v49 = sub_BD5D20(v47);
              LOWORD(v98) = 261;
              v96[0] = v50;
              v95 = v49;
              sub_2644DA0((__int64 *)&v89, v86, v50, 261, v51, v52, a7);
              v53 = sub_BA8DA0(a4, (__int64)v89, v90);
              v95 = (const char *)&v89;
              LOWORD(v98) = 260;
              v54 = v53;
              v55 = *(_QWORD *)(v47 + 8);
              if ( (unsigned int)*(unsigned __int8 *)(v55 + 8) - 17 <= 1 )
                v55 = **(_QWORD **)(v55 + 16);
              v56 = (unsigned __int8 *)sub_B30590(
                                         *(_QWORD **)(v47 + 24),
                                         *(_DWORD *)(v55 + 8) >> 8,
                                         *(_BYTE *)(v47 + 32) & 0xF,
                                         (__int64)&v95,
                                         (__int64)v83);
              sub_B31710((__int64)v56, (_BYTE *)v47);
              if ( v54 )
              {
                sub_BD6B90(v56, v54);
                sub_BD84D0((__int64)v54, (__int64)v56);
                sub_B30340(v54);
              }
              if ( v89 != &v91 )
                j_j___libc_free_0((unsigned __int64)v89);
              v57 = v48 + 1;
              if ( v48 + 1 == v46 )
                break;
              v47 = *v57;
              for ( ++v48; (unsigned __int64)*v57 >= 0xFFFFFFFFFFFFFFFELL; v48 = v57 )
              {
                if ( v46 == ++v57 )
                  goto LABEL_59;
                v47 = *v57;
              }
            }
            while ( v48 != v46 );
          }
        }
LABEL_59:
        if ( (__int64 *)v87[0] != &v88 )
          j_j___libc_free_0(v87[0]);
      }
      else
      {
LABEL_39:
        sub_2240A30((unsigned __int64 *)v87);
      }
      ++v86;
    }
    while ( a3 != v86 );
  }
  return a1;
}
