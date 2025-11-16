// Function: sub_1156740
// Address: 0x1156740
//
__int64 __fastcall sub_1156740(__int64 a1, __int64 a2, int a3, unsigned __int8 a4, unsigned __int8 a5)
{
  bool v9; // al
  __int64 result; // rax
  __int64 v11; // rax
  int *v12; // rax
  int v13; // eax
  unsigned __int8 v14; // al
  unsigned int v15; // r15d
  unsigned __int8 v16; // cl
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 *v19; // r13
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // r15
  __int64 v25; // rdx
  _BYTE *v26; // rax
  __int64 v27; // r12
  __int64 v28; // rcx
  unsigned int **v29; // rdi
  __int64 v30; // rsi
  unsigned int i; // r15d
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // r12
  __int64 v36; // rax
  unsigned int v37; // eax
  __int64 v38; // rcx
  int v39; // edi
  __int64 v40; // rcx
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // r12
  __int64 *v45; // r14
  __int64 v46; // r13
  _QWORD *v47; // rax
  __int64 v48; // rbx
  __int64 v49; // r12
  __int64 v50; // rdx
  unsigned int v51; // esi
  __int64 v52; // rsi
  __int64 v53; // r12
  __int64 v54; // r10
  __int64 *v55; // r13
  __int64 v56; // rbx
  __int64 v57; // r12
  __int64 v58; // rdx
  unsigned int v59; // esi
  __int64 v60; // rsi
  unsigned __int8 v61; // bl
  __int64 *v62; // r14
  __int64 v63; // r13
  char v64; // bl
  unsigned __int8 *v65; // rax
  __int64 v66; // rbx
  __int64 v67; // r12
  __int64 v68; // rdx
  unsigned int v69; // esi
  char v70; // [rsp+Fh] [rbp-A1h]
  __int64 v71; // [rsp+10h] [rbp-A0h]
  int v72; // [rsp+10h] [rbp-A0h]
  __int64 v74; // [rsp+18h] [rbp-98h]
  __int64 v75; // [rsp+18h] [rbp-98h]
  __int64 v76; // [rsp+18h] [rbp-98h]
  _QWORD *v77; // [rsp+18h] [rbp-98h]
  _QWORD *v78; // [rsp+18h] [rbp-98h]
  __int64 v79; // [rsp+18h] [rbp-98h]
  __int64 v80; // [rsp+18h] [rbp-98h]
  __int64 v81; // [rsp+18h] [rbp-98h]
  __int64 v82; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v83; // [rsp+18h] [rbp-98h]
  __int64 v84; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v85; // [rsp+18h] [rbp-98h]
  __int64 v86[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v87; // [rsp+40h] [rbp-70h]
  _BYTE v88[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v89; // [rsp+70h] [rbp-40h]

  if ( *(_BYTE *)a2 == 17 )
  {
    if ( *(_DWORD *)(a2 + 32) > 0x40u )
    {
      v9 = (unsigned int)sub_C44630(a2 + 24) == 1;
      goto LABEL_4;
    }
    v11 = *(_QWORD *)(a2 + 24);
    if ( !v11 )
      goto LABEL_9;
    goto LABEL_34;
  }
  v24 = *(_QWORD *)(a2 + 8);
  v25 = (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17;
  if ( (unsigned int)v25 > 1 || *(_BYTE *)a2 > 0x15u )
    goto LABEL_9;
  v26 = sub_AD7630(a2, 0, v25);
  if ( !v26 || *v26 != 17 )
  {
    if ( *(_BYTE *)(v24 + 8) != 17 )
      goto LABEL_9;
    v72 = *(_DWORD *)(v24 + 32);
    if ( !v72 )
      goto LABEL_9;
    v70 = 0;
    for ( i = 0; i != v72; ++i )
    {
      v32 = sub_AD69F0((unsigned __int8 *)a2, i);
      if ( !v32 )
        goto LABEL_9;
      if ( *(_BYTE *)v32 != 13 )
      {
        if ( *(_BYTE *)v32 != 17 )
          goto LABEL_9;
        if ( *(_DWORD *)(v32 + 32) > 0x40u )
        {
          if ( (unsigned int)sub_C44630(v32 + 24) != 1 )
            goto LABEL_9;
        }
        else
        {
          v33 = *(_QWORD *)(v32 + 24);
          if ( !v33 || (v33 & (v33 - 1)) != 0 )
            goto LABEL_9;
        }
        v70 = 1;
      }
    }
    if ( !v70 )
      goto LABEL_9;
LABEL_5:
    if ( a5 )
    {
      result = (__int64)sub_AD8AC0(a2);
      if ( !result )
        BUG();
      return result;
    }
    return -1;
  }
  if ( *((_DWORD *)v26 + 8) <= 0x40u )
  {
    v11 = *((_QWORD *)v26 + 3);
    if ( !v11 )
      goto LABEL_9;
LABEL_34:
    if ( (v11 & (v11 - 1)) != 0 )
      goto LABEL_9;
    goto LABEL_5;
  }
  v9 = (unsigned int)sub_C44630((__int64)(v26 + 24)) == 1;
LABEL_4:
  if ( v9 )
    goto LABEL_5;
LABEL_9:
  v12 = (int *)sub_C94E20((__int64)qword_4F862D0);
  if ( v12 )
    v13 = *v12;
  else
    v13 = qword_4F862D0[2];
  if ( a3 == v13 )
    return 0;
  v14 = *(_BYTE *)a2;
  v15 = a3 + 1;
  v16 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
    goto LABEL_37;
  if ( v14 == 68 )
  {
    v43 = *(_QWORD *)(a2 - 32);
    if ( !v43 )
      goto LABEL_38;
    v44 = sub_1156740(a1, v43, v15, a4, a5);
    if ( v44 )
    {
      result = -1;
      if ( !a5 )
        return result;
      v45 = *(__int64 **)(a1 + 32);
      v46 = *(_QWORD *)(a2 + 8);
      v87 = 257;
      if ( v46 != *(_QWORD *)(v44 + 8) )
      {
        result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v45[10] + 120LL))(
                   v45[10],
                   39,
                   v44,
                   v46);
        if ( !result )
        {
          v89 = 257;
          v47 = sub_BD2C40(72, unk_3F10A14);
          if ( v47 )
          {
            v77 = v47;
            sub_B515B0((__int64)v47, v44, v46, (__int64)v88, 0, 0);
            v47 = v77;
          }
          v78 = v47;
          (*(void (__fastcall **)(__int64, _QWORD *, __int64 *, __int64, __int64))(*(_QWORD *)v45[11] + 16LL))(
            v45[11],
            v47,
            v86,
            v45[7],
            v45[8]);
          v48 = *v45;
          result = (__int64)v78;
          v49 = *v45 + 16LL * *((unsigned int *)v45 + 2);
          if ( *v45 != v49 )
          {
            do
            {
              v50 = *(_QWORD *)(v48 + 8);
              v51 = *(_DWORD *)v48;
              v48 += 16;
              v79 = result;
              sub_B99FD0(result, v51, v50);
              result = v79;
            }
            while ( v49 != v48 );
          }
        }
        return result;
      }
      return v44;
    }
    v14 = *(_BYTE *)a2;
    v16 = *(_BYTE *)a2;
    if ( *(_BYTE *)a2 <= 0x1Cu )
      goto LABEL_37;
  }
  v16 = v14;
  if ( v14 != 67 )
    goto LABEL_15;
  v60 = *(_QWORD *)(a2 - 32);
  if ( !v60 )
    goto LABEL_38;
  if ( a4 || (*(_BYTE *)(a2 + 1) & 2) != 0 )
  {
    v44 = sub_1156740(a1, v60, v15, a4, a5);
    if ( v44 )
    {
      result = -1;
      if ( !a5 )
        return result;
      v61 = *(_BYTE *)(a2 + 1);
      v62 = *(__int64 **)(a1 + 32);
      v87 = 257;
      v63 = *(_QWORD *)(a2 + 8);
      v64 = v61 >> 1;
      if ( v63 != *(_QWORD *)(v44 + 8) )
      {
        result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v62[10] + 120LL))(
                   v62[10],
                   38,
                   v44,
                   v63);
        if ( !result )
        {
          v89 = 257;
          v65 = (unsigned __int8 *)sub_B51D30(38, v44, v63, (__int64)v88, 0, 0);
          if ( (v64 & 1) != 0 )
          {
            v85 = v65;
            sub_B447F0(v65, 1);
            v65 = v85;
          }
          v83 = v65;
          (*(void (__fastcall **)(__int64, unsigned __int8 *, __int64 *, __int64, __int64))(*(_QWORD *)v62[11] + 16LL))(
            v62[11],
            v65,
            v86,
            v62[7],
            v62[8]);
          v66 = *v62;
          result = (__int64)v83;
          v67 = *v62 + 16LL * *((unsigned int *)v62 + 2);
          if ( *v62 != v67 )
          {
            do
            {
              v68 = *(_QWORD *)(v66 + 8);
              v69 = *(_DWORD *)v66;
              v66 += 16;
              v84 = result;
              sub_B99FD0(result, v69, v68);
              result = v84;
            }
            while ( v67 != v66 );
          }
        }
        return result;
      }
      return v44;
    }
    v14 = *(_BYTE *)a2;
  }
  v16 = v14;
LABEL_15:
  if ( v14 == 54 )
  {
    v17 = *(_QWORD *)(a2 - 64);
    if ( !v17 || !*(_QWORD *)(a2 - 32) || !a4 && (*(_BYTE *)(a2 + 1) & 2) == 0 && ((*(_BYTE *)(a2 + 1) >> 1) & 2) == 0 )
      return 0;
    v71 = *(_QWORD *)(a2 - 32);
    v18 = sub_1156740(a1, v17, v15, a4, a5);
    if ( v18 )
    {
      result = -1;
      if ( a5 )
      {
        v19 = *(__int64 **)(a1 + 32);
        v87 = 257;
        result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v19[10]
                                                                                               + 32LL))(
                   v19[10],
                   13,
                   v18,
                   v71,
                   0,
                   0);
        if ( !result )
        {
          v89 = 257;
          v74 = sub_B504D0(13, v18, v71, (__int64)v88, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v19[11] + 16LL))(
            v19[11],
            v74,
            v86,
            v19[7],
            v19[8]);
          v20 = *v19;
          result = v74;
          v21 = *v19 + 16LL * *((unsigned int *)v19 + 2);
          if ( *v19 != v21 )
          {
            do
            {
              v22 = *(_QWORD *)(v20 + 8);
              v23 = *(_DWORD *)v20;
              v20 += 16;
              v75 = result;
              sub_B99FD0(result, v23, v22);
              result = v75;
            }
            while ( v21 != v20 );
          }
        }
      }
      return result;
    }
    v16 = *(_BYTE *)a2;
  }
LABEL_37:
  if ( v16 == 55 )
  {
    v52 = *(_QWORD *)(a2 - 64);
    if ( !v52 )
      return 0;
    v53 = *(_QWORD *)(a2 - 32);
    if ( !v53 || !a4 && (*(_BYTE *)(a2 + 1) & 2) == 0 )
      return 0;
    v54 = sub_1156740(a1, v52, v15, a4, a5);
    if ( v54 )
    {
      result = -1;
      if ( a5 )
      {
        v55 = *(__int64 **)(a1 + 32);
        v87 = 257;
        v80 = v54;
        result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v55[10]
                                                                                               + 32LL))(
                   v55[10],
                   15,
                   v54,
                   v53,
                   0,
                   0);
        if ( !result )
        {
          v89 = 257;
          v81 = sub_B504D0(15, v80, v53, (__int64)v88, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v55[11] + 16LL))(
            v55[11],
            v81,
            v86,
            v55[7],
            v55[8]);
          v56 = *v55;
          result = v81;
          v57 = *v55 + 16LL * *((unsigned int *)v55 + 2);
          if ( *v55 != v57 )
          {
            do
            {
              v58 = *(_QWORD *)(v56 + 8);
              v59 = *(_DWORD *)v56;
              v56 += 16;
              v82 = result;
              sub_B99FD0(result, v59, v58);
              result = v82;
            }
            while ( v57 != v56 );
          }
        }
      }
      return result;
    }
    v16 = *(_BYTE *)a2;
  }
LABEL_38:
  if ( a4 && v16 == 57 )
  {
    v34 = *(_QWORD *)(a2 - 64);
    if ( !v34 )
      return 0;
    v35 = *(_QWORD *)(a2 - 32);
    if ( !v35 )
      return 0;
    result = sub_1156740(a1, v34, v15, 1, a5);
    if ( result || (result = sub_1156740(a1, v35, v15, 1, a5)) != 0 )
    {
      if ( !a5 )
        return -1;
      return result;
    }
    v16 = *(_BYTE *)a2;
  }
  if ( v16 == 86 )
  {
    v27 = sub_1156740(a1, *(_QWORD *)(a2 - 64), v15, a4, a5);
    if ( v27 )
    {
      v28 = sub_1156740(a1, *(_QWORD *)(a2 - 32), v15, a4, a5);
      if ( v28 )
      {
        result = -1;
        if ( a5 )
        {
          v29 = *(unsigned int ***)(a1 + 32);
          v30 = *(_QWORD *)(a2 - 96);
          v89 = 257;
          return sub_B36550(v29, v30, v27, v28, (__int64)v88, 0);
        }
        return result;
      }
    }
    v16 = *(_BYTE *)a2;
  }
  if ( v16 != 85 )
    return 0;
  v36 = *(_QWORD *)(a2 - 32);
  if ( !v36 || *(_BYTE *)v36 || *(_QWORD *)(v36 + 24) != *(_QWORD *)(a2 + 80) || (*(_BYTE *)(v36 + 33) & 0x20) == 0 )
    return 0;
  v37 = *(_DWORD *)(v36 + 36);
  if ( v37 <= 0x14A )
  {
    if ( v37 > 0x148 )
      goto LABEL_77;
    return 0;
  }
  if ( v37 - 365 > 1 )
    return 0;
LABEL_77:
  v38 = *(_QWORD *)(a2 + 16);
  if ( !v38 || *(_QWORD *)(v38 + 8) )
    return 0;
  if ( v37 == 365 )
  {
    v39 = 34;
    goto LABEL_83;
  }
  if ( v37 > 0x16D )
  {
    if ( v37 == 366 )
    {
      v39 = 36;
      goto LABEL_83;
    }
    goto LABEL_138;
  }
  if ( v37 != 329 )
  {
    if ( v37 == 330 )
    {
      v39 = 40;
      goto LABEL_83;
    }
LABEL_138:
    BUG();
  }
  v39 = 38;
LABEL_83:
  if ( sub_B532B0(v39) )
    return 0;
  v76 = sub_1156740(a1, *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v15, 0, a5);
  if ( !v76 )
    return 0;
  v40 = sub_1156740(a1, *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), v15, 0, a5);
  if ( !v40 )
    return 0;
  if ( !a5 )
    return -1;
  v41 = *(_QWORD *)(a1 + 32);
  HIDWORD(v86[0]) = 0;
  v89 = 257;
  v42 = *(_QWORD *)(a2 - 32);
  if ( !v42 || *(_BYTE *)v42 || *(_QWORD *)(v42 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  return sub_B33C40(v41, *(_DWORD *)(v42 + 36), v76, v40, v86[0], (__int64)v88);
}
