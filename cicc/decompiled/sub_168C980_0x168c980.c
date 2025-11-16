// Function: sub_168C980
// Address: 0x168c980
//
_QWORD *__fastcall sub_168C980(__int64 a1, void (__fastcall *a2)(__int64, void **), __int64 a3)
{
  _QWORD *result; // rax
  __int64 v4; // rbx
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 (__fastcall *v8)(_QWORD *, _QWORD *); // r14
  __int64 v9; // r14
  __int64 (__fastcall *v10)(__int64, _QWORD *); // r15
  __int64 (__fastcall *v11)(_QWORD *, const char *, _QWORD, const char *, _QWORD); // r15
  _QWORD *v12; // r15
  __int64 (*v13)(void); // rax
  void (__fastcall *v14)(void **); // rax
  _QWORD *v15; // rsi
  __int64 v16; // r12
  __int64 (__fastcall *v17)(_QWORD *, __int64, __int64, __int64 *); // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  void **v20; // rsi
  _QWORD *v21; // rbx
  _QWORD *v22; // r8
  _QWORD *v23; // r12
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  __int64 v26; // rbx
  __int64 v27; // r12
  __int64 v28; // rdi
  _QWORD *v29; // rbx
  _QWORD *v30; // r12
  __int64 v31; // rdi
  unsigned __int64 v32; // r8
  __int64 (__fastcall *v33)(_QWORD *); // rax
  _QWORD *v34; // rdi
  _QWORD *v35; // rdi
  _QWORD *v36; // rbx
  _QWORD *v37; // r8
  _QWORD *v38; // r12
  _QWORD *v39; // rbx
  _QWORD *v40; // r12
  __int64 v41; // rbx
  __int64 v42; // r12
  __int64 v43; // rdi
  _QWORD *v44; // rbx
  _QWORD *v45; // r12
  __int64 v46; // rdi
  __int64 v47; // r12
  __int64 v48; // rbx
  unsigned __int64 v49; // rdi
  __int64 v50; // r12
  __int64 v51; // rbx
  unsigned __int64 v52; // rdi
  __int64 v53; // [rsp+18h] [rbp-BB8h]
  __int64 v56; // [rsp+38h] [rbp-B98h]
  _QWORD *v57; // [rsp+40h] [rbp-B90h]
  _QWORD *v58; // [rsp+58h] [rbp-B78h]
  __int64 v59; // [rsp+58h] [rbp-B78h]
  __int64 v60; // [rsp+58h] [rbp-B78h]
  __int64 v61; // [rsp+68h] [rbp-B68h] BYREF
  _QWORD v62[2]; // [rsp+70h] [rbp-B60h] BYREF
  _QWORD v63[2]; // [rsp+80h] [rbp-B50h] BYREF
  _QWORD *v64; // [rsp+90h] [rbp-B40h] BYREF
  __int64 v65; // [rsp+98h] [rbp-B38h]
  _QWORD v66[6]; // [rsp+A0h] [rbp-B30h] BYREF
  __int64 v67; // [rsp+D0h] [rbp-B00h] BYREF
  __int64 v68; // [rsp+D8h] [rbp-AF8h]
  __int64 v69; // [rsp+E0h] [rbp-AF0h]
  _QWORD *v70; // [rsp+E8h] [rbp-AE8h]
  _QWORD *v71; // [rsp+F0h] [rbp-AE0h]
  __int64 v72; // [rsp+F8h] [rbp-AD8h]
  __int64 v73; // [rsp+100h] [rbp-AD0h]
  __int64 v74; // [rsp+108h] [rbp-AC8h]
  __int64 v75; // [rsp+110h] [rbp-AC0h] BYREF
  _QWORD *v76; // [rsp+118h] [rbp-AB8h]
  __int64 v77; // [rsp+120h] [rbp-AB0h]
  _QWORD v78[2]; // [rsp+128h] [rbp-AA8h] BYREF
  _QWORD *v79; // [rsp+138h] [rbp-A98h]
  _QWORD v80[2]; // [rsp+148h] [rbp-A88h] BYREF
  _QWORD *v81; // [rsp+158h] [rbp-A78h]
  _QWORD *v82; // [rsp+160h] [rbp-A70h]
  __int64 v83; // [rsp+168h] [rbp-A68h]
  void *v84; // [rsp+170h] [rbp-A60h] BYREF
  __int64 v85; // [rsp+178h] [rbp-A58h]
  unsigned __int64 v86; // [rsp+280h] [rbp-950h]
  unsigned int v87; // [rsp+288h] [rbp-948h]
  int v88; // [rsp+28Ch] [rbp-944h]
  _QWORD *v89; // [rsp+2A8h] [rbp-928h]
  unsigned int v90; // [rsp+2B8h] [rbp-918h]
  _QWORD v91[2]; // [rsp+2C0h] [rbp-910h] BYREF
  __int16 v92; // [rsp+2D0h] [rbp-900h]
  __int64 v93; // [rsp+468h] [rbp-768h]
  __int64 v94; // [rsp+470h] [rbp-760h]
  __int64 v95; // [rsp+478h] [rbp-758h]
  int v96; // [rsp+480h] [rbp-750h]
  _QWORD *v97; // [rsp+578h] [rbp-658h]
  __int64 v98; // [rsp+580h] [rbp-650h]
  _QWORD v99[5]; // [rsp+588h] [rbp-648h] BYREF
  _QWORD v100[2]; // [rsp+5B0h] [rbp-620h] BYREF
  _QWORD v101[194]; // [rsp+5C0h] [rbp-610h] BYREF

  result = *(_QWORD **)(a1 + 88);
  v4 = *(_QWORD *)(a1 + 96);
  v57 = result;
  if ( !v4 )
    return result;
  v62[1] = 0;
  v62[0] = v63;
  v100[0] = a1 + 240;
  LOWORD(v101[0]) = 260;
  LOBYTE(v63[0]) = 0;
  sub_16E1010(&v64);
  v6 = sub_16D3AC0(&v64, v62);
  v7 = v64;
  v8 = *(__int64 (__fastcall **)(_QWORD *, _QWORD *))(v6 + 72);
  v58 = (_QWORD *)v6;
  v84 = v64;
  v85 = v65;
  if ( v8 )
  {
    v91[0] = &v84;
    v92 = 261;
    sub_16E1010(v100);
    v9 = v8(v100, v91);
    if ( (_QWORD *)v100[0] != v101 )
      j_j___libc_free_0(v100[0], v101[0] + 1LL);
    v7 = v64;
    if ( v9 )
    {
      v10 = (__int64 (__fastcall *)(__int64, _QWORD *))v58[6];
      v84 = v64;
      v85 = v65;
      if ( !v10 )
        goto LABEL_67;
      v92 = 261;
      v91[0] = &v84;
      sub_16E1010(v100);
      v56 = v10(v9, v100);
      if ( (_QWORD *)v100[0] != v101 )
        j_j___libc_free_0(v100[0], v101[0] + 1LL);
      if ( !v56 )
        goto LABEL_67;
      v11 = (__int64 (__fastcall *)(_QWORD *, const char *, _QWORD, const char *, _QWORD))v58[10];
      v84 = v64;
      v85 = v65;
      if ( !v11 )
        goto LABEL_66;
      v92 = 261;
      v91[0] = &v84;
      sub_16E1010(v100);
      v12 = (_QWORD *)v11(v100, byte_3F871B3, 0, byte_3F871B3, 0);
      if ( (_QWORD *)v100[0] != v101 )
        j_j___libc_free_0(v100[0], v101[0] + 1LL);
      if ( !v12 )
      {
LABEL_66:
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v56 + 8LL))(v56);
LABEL_67:
        j___libc_free_0(*(_QWORD *)(v9 + 200));
        j___libc_free_0(*(_QWORD *)(v9 + 168));
        j_j___libc_free_0(v9, 224);
        v7 = v64;
        goto LABEL_68;
      }
      v13 = (__int64 (*)(void))v58[7];
      if ( !v13 || (v53 = v13()) == 0 )
      {
LABEL_60:
        v33 = *(__int64 (__fastcall **)(_QWORD *))(*v12 + 8LL);
        if ( v33 == sub_168C4D0 )
        {
          v34 = (_QWORD *)v12[8];
          *v12 = &unk_49EE580;
          if ( v34 != v12 + 10 )
            j_j___libc_free_0(v34, v12[10] + 1LL);
          v35 = (_QWORD *)v12[1];
          if ( v35 != v12 + 3 )
            j_j___libc_free_0(v35, v12[3] + 1LL);
          j_j___libc_free_0(v12, 216);
        }
        else
        {
          v33(v12);
        }
        goto LABEL_66;
      }
      v97 = v99;
      v93 = 0;
      v94 = 0;
      v95 = 0;
      v96 = 0;
      v98 = 0;
      LOBYTE(v99[0]) = 0;
      memset(&v99[2], 0, 24);
      sub_38BCD70(v100, v56, v9, v91, 0, 1);
      sub_38D34B0(v91, &v64, 0, v100, 0);
      sub_168E6A0(&v84, v100, a1);
      v14 = (void (__fastcall *)(void **))v58[22];
      if ( v14 )
        v14(&v84);
      sub_16C2450(&v61, v57, v4, byte_3F871B3, 0, 1);
      v67 = 0;
      v75 = v61;
      v68 = 0;
      v69 = 0;
      v70 = 0;
      v71 = 0;
      v72 = 0;
      v73 = 0;
      v74 = 0;
      v61 = 0;
      v76 = 0;
      v77 = 0;
      sub_168C7C0(&v67, 0, (__int64)&v75);
      sub_16CE300(&v75, 0);
      v15 = v100;
      v16 = sub_38E8880(&v67, v100, &v84, v56, 0);
      sub_167F890((__int16 *)&v75);
      v17 = (__int64 (__fastcall *)(_QWORD *, __int64, __int64, __int64 *))v58[13];
      if ( v17 )
      {
        v15 = (_QWORD *)v16;
        v18 = v17(v12, v16, v53, &v75);
        v19 = v18;
        if ( v18 )
        {
          sub_3909440(v16, v18);
          v15 = 0;
          if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v16 + 80LL))(v16, 0, 0) )
          {
            v20 = &v84;
            a2(a3, &v84);
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
            v21 = v82;
            v22 = v81;
            if ( v82 != v81 )
            {
              v59 = v16;
              v23 = v81;
              do
              {
                if ( (_QWORD *)*v23 != v23 + 2 )
                {
                  v20 = (void **)(v23[2] + 1LL);
                  j_j___libc_free_0(*v23, v20);
                }
                v23 += 4;
              }
              while ( v21 != v23 );
              v16 = v59;
              v22 = v81;
            }
            if ( v22 )
            {
              v20 = (void **)(v83 - (_QWORD)v22);
              j_j___libc_free_0(v22, v83 - (_QWORD)v22);
            }
            if ( v79 != v80 )
            {
              v20 = (void **)(v80[0] + 1LL);
              j_j___libc_free_0(v79, v80[0] + 1LL);
            }
            if ( v76 != v78 )
            {
              v20 = (void **)(v78[0] + 1LL);
              j_j___libc_free_0(v76, v78[0] + 1LL);
            }
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
            v24 = v71;
            v25 = v70;
            if ( v71 != v70 )
            {
              do
              {
                if ( (_QWORD *)*v25 != v25 + 2 )
                {
                  v20 = (void **)(v25[2] + 1LL);
                  j_j___libc_free_0(*v25, v20);
                }
                v25 += 4;
              }
              while ( v24 != v25 );
              v25 = v70;
            }
            if ( v25 )
            {
              v20 = (void **)(v72 - (_QWORD)v25);
              j_j___libc_free_0(v25, v72 - (_QWORD)v25);
            }
            v26 = v68;
            v27 = v67;
            if ( v68 != v67 )
            {
              do
              {
                v28 = v27;
                v27 += 24;
                sub_16CE300(v28, v20);
              }
              while ( v26 != v27 );
              v27 = v67;
            }
            if ( v27 )
              j_j___libc_free_0(v27, v69 - v27);
            if ( v61 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v61 + 8LL))(v61);
            v84 = &unk_49EE5B0;
            if ( v90 )
            {
              v29 = v89;
              v30 = &v89[4 * v90];
              do
              {
                if ( *v29 != -16 && *v29 != -8 )
                {
                  v31 = v29[1];
                  if ( v31 )
                    j_j___libc_free_0(v31, v29[3] - v31);
                }
                v29 += 4;
              }
              while ( v30 != v29 );
            }
            j___libc_free_0(v89);
            if ( v88 )
            {
              v32 = v86;
              if ( v87 )
              {
                v50 = 8LL * v87;
                v51 = 0;
                do
                {
                  v52 = *(_QWORD *)(v32 + v51);
                  if ( v52 && v52 != -8 )
                  {
                    _libc_free(v52);
                    v32 = v86;
                  }
                  v51 += 8;
                }
                while ( v50 != v51 );
              }
              goto LABEL_57;
            }
LABEL_56:
            v32 = v86;
LABEL_57:
            _libc_free(v32);
            sub_38DCBC0(&v84);
            sub_38C0FE0(v100);
            if ( v97 != v99 )
              j_j___libc_free_0(v97, v99[0] + 1LL);
            j___libc_free_0(v94);
            j_j___libc_free_0(v53, 32);
            goto LABEL_60;
          }
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v19 + 8LL))(v19);
        }
      }
      v36 = v82;
      v37 = v81;
      if ( v82 != v81 )
      {
        v60 = v16;
        v38 = v81;
        do
        {
          if ( (_QWORD *)*v38 != v38 + 2 )
          {
            v15 = (_QWORD *)(v38[2] + 1LL);
            j_j___libc_free_0(*v38, v15);
          }
          v38 += 4;
        }
        while ( v36 != v38 );
        v16 = v60;
        v37 = v81;
      }
      if ( v37 )
      {
        v15 = (_QWORD *)(v83 - (_QWORD)v37);
        j_j___libc_free_0(v37, v83 - (_QWORD)v37);
      }
      if ( v79 != v80 )
      {
        v15 = (_QWORD *)(v80[0] + 1LL);
        j_j___libc_free_0(v79, v80[0] + 1LL);
      }
      if ( v76 != v78 )
      {
        v15 = (_QWORD *)(v78[0] + 1LL);
        j_j___libc_free_0(v76, v78[0] + 1LL);
      }
      if ( v16 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
      v39 = v71;
      v40 = v70;
      if ( v71 != v70 )
      {
        do
        {
          if ( (_QWORD *)*v40 != v40 + 2 )
          {
            v15 = (_QWORD *)(v40[2] + 1LL);
            j_j___libc_free_0(*v40, v15);
          }
          v40 += 4;
        }
        while ( v39 != v40 );
        v40 = v70;
      }
      if ( v40 )
      {
        v15 = (_QWORD *)(v72 - (_QWORD)v40);
        j_j___libc_free_0(v40, v72 - (_QWORD)v40);
      }
      v41 = v68;
      v42 = v67;
      if ( v68 != v67 )
      {
        do
        {
          v43 = v42;
          v42 += 24;
          sub_16CE300(v43, v15);
        }
        while ( v41 != v42 );
        v42 = v67;
      }
      if ( v42 )
        j_j___libc_free_0(v42, v69 - v42);
      if ( v61 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v61 + 8LL))(v61);
      v84 = &unk_49EE5B0;
      if ( v90 )
      {
        v44 = v89;
        v45 = &v89[4 * v90];
        do
        {
          if ( *v44 != -16 && *v44 != -8 )
          {
            v46 = v44[1];
            if ( v46 )
              j_j___libc_free_0(v46, v44[3] - v46);
          }
          v44 += 4;
        }
        while ( v45 != v44 );
      }
      j___libc_free_0(v89);
      if ( v88 )
      {
        v32 = v86;
        if ( v87 )
        {
          v47 = 8LL * v87;
          v48 = 0;
          do
          {
            v49 = *(_QWORD *)(v32 + v48);
            if ( v49 && v49 != -8 )
            {
              _libc_free(v49);
              v32 = v86;
            }
            v48 += 8;
          }
          while ( v47 != v48 );
        }
        goto LABEL_57;
      }
      goto LABEL_56;
    }
  }
LABEL_68:
  result = v66;
  if ( v7 != v66 )
    result = (_QWORD *)j_j___libc_free_0(v7, v66[0] + 1LL);
  if ( (_QWORD *)v62[0] != v63 )
    return (_QWORD *)j_j___libc_free_0(v62[0], v63[0] + 1LL);
  return result;
}
