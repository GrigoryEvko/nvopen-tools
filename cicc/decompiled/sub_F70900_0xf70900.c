// Function: sub_F70900
// Address: 0xf70900
//
__int64 __fastcall sub_F70900(unsigned __int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rbx
  _QWORD *v9; // r12
  unsigned __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 **v13; // rax
  __int64 result; // rax
  _QWORD *v15; // r14
  _QWORD *v16; // r12
  void (__fastcall *v17)(_QWORD *, _QWORD *, __int64); // rax
  unsigned int v18; // ebx
  __int64 **v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // rdx
  unsigned __int64 *v25; // rax
  __int64 **v26; // rax
  __int64 *v27; // rcx
  unsigned __int64 v28; // rax
  int v29; // edx
  __int64 v30; // rdi
  __int64 v31; // rax
  _QWORD *v32; // r14
  void (__fastcall *v33)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v34; // r13
  int v35; // eax
  int v36; // r15d
  __int64 v37; // rsi
  _QWORD *v38; // rax
  _QWORD *v39; // rdx
  unsigned __int64 *v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // rsi
  _QWORD *v43; // rax
  _QWORD *v44; // rdx
  __int64 v45; // rax
  __int64 v46; // r13
  unsigned __int64 *v47; // rax
  __int64 *v48; // rcx
  __int64 *v49; // rax
  _QWORD *v50; // rax
  __int64 v51; // r15
  __int64 v52; // r13
  _BYTE *v53; // r13
  _BYTE *v54; // rbx
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int64 v57; // rax
  __int64 v58; // rcx
  __int64 *v59; // rdi
  __int64 v60; // rax
  _QWORD *v61; // r14
  void (__fastcall *v62)(_QWORD *, _QWORD *, __int64); // rax
  unsigned __int64 v63; // [rsp+28h] [rbp-3B8h]
  unsigned __int64 v64; // [rsp+30h] [rbp-3B0h] BYREF
  __int64 v65; // [rsp+38h] [rbp-3A8h]
  __int16 v66; // [rsp+50h] [rbp-390h]
  _BYTE *v67; // [rsp+60h] [rbp-380h] BYREF
  __int64 v68; // [rsp+68h] [rbp-378h]
  _BYTE v69[32]; // [rsp+70h] [rbp-370h] BYREF
  __int64 v70; // [rsp+90h] [rbp-350h]
  __int64 v71; // [rsp+98h] [rbp-348h]
  __int64 v72; // [rsp+A0h] [rbp-340h]
  __int64 v73; // [rsp+A8h] [rbp-338h]
  void **v74; // [rsp+B0h] [rbp-330h]
  void **v75; // [rsp+B8h] [rbp-328h]
  __int64 v76; // [rsp+C0h] [rbp-320h]
  int v77; // [rsp+C8h] [rbp-318h]
  __int16 v78; // [rsp+CCh] [rbp-314h]
  char v79; // [rsp+CEh] [rbp-312h]
  __int64 v80; // [rsp+D0h] [rbp-310h]
  __int64 v81; // [rsp+D8h] [rbp-308h]
  void *v82; // [rsp+E0h] [rbp-300h] BYREF
  void *v83; // [rsp+E8h] [rbp-2F8h] BYREF
  _BYTE *v84; // [rsp+F0h] [rbp-2F0h] BYREF
  __int64 v85; // [rsp+F8h] [rbp-2E8h]
  _BYTE v86[16]; // [rsp+100h] [rbp-2E0h] BYREF
  __int16 v87; // [rsp+110h] [rbp-2D0h]
  __int64 v88; // [rsp+300h] [rbp-E0h]
  __int64 v89; // [rsp+308h] [rbp-D8h]
  unsigned __int64 *v90; // [rsp+310h] [rbp-D0h]
  __int64 v91; // [rsp+318h] [rbp-C8h]
  char v92; // [rsp+320h] [rbp-C0h]
  __int64 v93; // [rsp+328h] [rbp-B8h]
  _BYTE *v94; // [rsp+330h] [rbp-B0h]
  __int64 v95; // [rsp+338h] [rbp-A8h]
  int v96; // [rsp+340h] [rbp-A0h]
  char v97; // [rsp+344h] [rbp-9Ch]
  _BYTE v98[64]; // [rsp+348h] [rbp-98h] BYREF
  __int16 v99; // [rsp+388h] [rbp-58h]
  _QWORD *v100; // [rsp+390h] [rbp-50h]
  _QWORD *v101; // [rsp+398h] [rbp-48h]
  __int64 v102; // [rsp+3A0h] [rbp-40h]

  v6 = (__int64 *)*a1;
  v7 = **a1;
  v8 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 == v7 + 48 )
    goto LABEL_70;
  if ( !v8 )
    BUG();
  v9 = (_QWORD *)(v8 - 24);
  if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
LABEL_70:
    BUG();
  if ( *(_BYTE *)(v8 - 24) == 31 )
  {
    if ( (*(_DWORD *)(v8 - 20) & 0x7FFFFFF) != 3 )
    {
      v11 = a1[1];
      v99 = 0;
      v12 = 1;
      v90 = v11;
      v94 = v98;
      v13 = (__int64 **)a1[2];
      v84 = v86;
      v85 = 0x1000000000LL;
      v88 = 0;
      v89 = 0;
      v91 = 0;
      v92 = 0;
      v93 = 0;
      v95 = 8;
      v96 = 0;
      v97 = 1;
      v100 = 0;
      v101 = 0;
      v102 = 0;
      sub_F55BE0((__int64)v9, 1u, (__int64)&v84, *v13, a5, a6);
      sub_FFCE90(&v84);
      sub_FFD870(&v84);
      result = sub_FFBC40(&v84);
      v15 = v101;
      v16 = v100;
      if ( v101 != v100 )
      {
        do
        {
          v17 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v16[7];
          *v16 = &unk_49E5048;
          if ( v17 )
          {
            v12 = (__int64)(v16 + 5);
            v17(v16 + 5, v16 + 5, 3);
          }
          *v16 = &unk_49DB368;
          result = v16[3];
          if ( result != -4096 && result != 0 && result != -8192 )
            result = sub_BD60C0(v16 + 1);
          v16 += 9;
        }
        while ( v15 != v16 );
        goto LABEL_13;
      }
      goto LABEL_14;
    }
    v34 = *a1[3];
    v35 = sub_B46E30(v8 - 24);
    if ( v35 )
    {
      v63 = v8;
      v18 = 0;
      v36 = v35;
      while ( 1 )
      {
        v37 = sub_B46EC0((__int64)v9, v18);
        if ( *(_BYTE *)(v34 + 84) )
        {
          v38 = *(_QWORD **)(v34 + 64);
          v39 = &v38[*(unsigned int *)(v34 + 76)];
          if ( v38 == v39 )
          {
LABEL_44:
            v40 = a1[3];
            v41 = *v40;
            v42 = *(_QWORD *)(v63 - 56);
            if ( *(_BYTE *)(*v40 + 84) )
            {
              v43 = *(_QWORD **)(v41 + 64);
              v44 = &v43[*(unsigned int *)(v41 + 76)];
              if ( v43 == v44 )
              {
LABEL_46:
                v45 = -32;
              }
              else
              {
                while ( v42 != *v43 )
                {
                  if ( v44 == ++v43 )
                    goto LABEL_46;
                }
                v45 = -64;
              }
            }
            else
            {
              v45 = -32 - 32LL * (sub_C8CA60(v41 + 56, v42) != 0);
            }
            v46 = *(_QWORD *)(v63 + v45 - 24);
            v47 = a1[1];
            v99 = 0;
            v90 = v47;
            v85 = 0x1000000000LL;
            v48 = (__int64 *)*a1;
            v94 = v98;
            v49 = (__int64 *)a1[4];
            v84 = v86;
            v88 = 0;
            v89 = 0;
            v91 = 0;
            v92 = 0;
            v93 = 0;
            v95 = 8;
            v96 = 0;
            v97 = 1;
            v100 = 0;
            v101 = 0;
            v102 = 0;
            sub_AA5980(*v49, *v48, 1u);
            v73 = sub_BD5C60((__int64)v9);
            v74 = &v82;
            v75 = &v83;
            v67 = v69;
            v82 = &unk_49DA100;
            v68 = 0x200000000LL;
            v78 = 512;
            LOWORD(v72) = 0;
            v83 = &unk_49DA0B0;
            v76 = 0;
            v77 = 0;
            v79 = 7;
            v80 = 0;
            v81 = 0;
            v70 = 0;
            v71 = 0;
            sub_D5F1F0((__int64)&v67, (__int64)v9);
            v66 = 257;
            v50 = sub_BD2C40(72, 1u);
            v51 = (__int64)v50;
            if ( v50 )
              sub_B4C8F0((__int64)v50, v46, 1u, 0, 0);
            (*((void (__fastcall **)(void **, __int64, unsigned __int64 *, __int64, __int64))*v75 + 2))(
              v75,
              v51,
              &v64,
              v71,
              v72);
            v52 = 16LL * (unsigned int)v68;
            if ( v67 != &v67[v52] )
            {
              v53 = &v67[v52];
              v54 = v67;
              do
              {
                v55 = *((_QWORD *)v54 + 1);
                v56 = *(_DWORD *)v54;
                v54 += 16;
                sub_B99FD0(v51, v56, v55);
              }
              while ( v53 != v54 );
            }
            v64 = 0x1E00000000LL;
            sub_B47C00(v51, (__int64)v9, (int *)&v64, 2);
            sub_B43D60(v9);
            v12 = (__int64)&v64;
            v57 = *a1[4];
            v64 = **a1;
            v65 = v57 | 4;
            sub_FFB3D0(&v84, &v64, 1);
            if ( *a1[5] )
            {
              v58 = (__int64)a1[1];
              v12 = (__int64)&v64;
              v59 = (__int64 *)*a1[2];
              v60 = *a1[4];
              v64 = **a1;
              v65 = v60 | 4;
              sub_D75690(v59, &v64, 1, v58, 0);
            }
            nullsub_61();
            v82 = &unk_49DA100;
            nullsub_63();
            if ( v67 != v69 )
              _libc_free(v67, &v64);
            sub_FFCE90(&v84);
            sub_FFD870(&v84);
            result = sub_FFBC40(&v84);
            v61 = v101;
            v16 = v100;
            if ( v101 != v100 )
            {
              do
              {
                v62 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v16[7];
                *v16 = &unk_49E5048;
                if ( v62 )
                {
                  v12 = (__int64)(v16 + 5);
                  v62(v16 + 5, v16 + 5, 3);
                }
                *v16 = &unk_49DB368;
                result = v16[3];
                if ( result != 0 && result != -4096 && result != -8192 )
                  result = sub_BD60C0(v16 + 1);
                v16 += 9;
              }
              while ( v61 != v16 );
              goto LABEL_13;
            }
            goto LABEL_14;
          }
          while ( v37 != *v38 )
          {
            if ( v39 == ++v38 )
              goto LABEL_44;
          }
        }
        else if ( !sub_C8CA60(v34 + 56, v37) )
        {
          goto LABEL_44;
        }
        if ( v36 == ++v18 )
        {
          v6 = (__int64 *)*a1;
          break;
        }
      }
    }
  }
  v19 = (__int64 **)a1[2];
  v20 = (__int64)a1[6];
  v87 = 257;
  v21 = sub_F41C30(*v6, *a1[4], (__int64)a1[1], v20, *v19, (void **)&v84);
  v84 = v86;
  v24 = (_QWORD *)v21;
  v25 = a1[1];
  v85 = 0x1000000000LL;
  v88 = 0;
  v24 += 6;
  v90 = v25;
  v94 = v98;
  v26 = (__int64 **)a1[2];
  v89 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v95 = 8;
  v96 = 0;
  v97 = 1;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v27 = *v26;
  v28 = *v24 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v28 == v24 )
  {
    v30 = 0;
  }
  else
  {
    if ( !v28 )
      BUG();
    v29 = *(unsigned __int8 *)(v28 - 24);
    v30 = 0;
    v31 = v28 - 24;
    if ( (unsigned int)(v29 - 30) < 0xB )
      v30 = v31;
  }
  v12 = 1;
  sub_F55BE0(v30, 1u, (__int64)&v84, v27, v22, v23);
  sub_FFCE90(&v84);
  sub_FFD870(&v84);
  result = sub_FFBC40(&v84);
  v32 = v101;
  v16 = v100;
  if ( v101 != v100 )
  {
    do
    {
      v33 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v16[7];
      *v16 = &unk_49E5048;
      if ( v33 )
      {
        v12 = (__int64)(v16 + 5);
        v33(v16 + 5, v16 + 5, 3);
      }
      *v16 = &unk_49DB368;
      result = v16[3];
      if ( result != 0 && result != -4096 && result != -8192 )
        result = sub_BD60C0(v16 + 1);
      v16 += 9;
    }
    while ( v32 != v16 );
LABEL_13:
    v16 = v100;
  }
LABEL_14:
  if ( v16 )
  {
    v12 = v102 - (_QWORD)v16;
    result = j_j___libc_free_0(v16, v102 - (_QWORD)v16);
  }
  if ( !v97 )
    result = _libc_free(v94, v12);
  if ( v84 != v86 )
    return _libc_free(v84, v12);
  return result;
}
