// Function: sub_1042620
// Address: 0x1042620
//
_QWORD *__fastcall sub_1042620(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax
  int v8; // r13d
  __int64 v9; // rsi
  _QWORD *v10; // r14
  int v11; // eax
  bool v12; // zf
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // r14
  __int64 *v17; // r13
  __int64 *v18; // rbx
  __int64 v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  int v31; // esi
  __int64 v32; // rdi
  __int64 v33; // rdx
  int v34; // esi
  unsigned int v35; // ecx
  __int64 *v36; // rax
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // rdi
  __int64 *v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 *v45; // r14
  char v46; // di
  __int64 *v47; // r15
  __int64 v48; // rsi
  __int64 *v49; // rax
  __int64 ***v50; // r14
  __int64 v51; // r15
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 *v54; // rsi
  _QWORD *result; // rax
  char v56; // cl
  __int64 v57; // r13
  __int64 v58; // r15
  __int64 *v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rsi
  __int64 v62; // rcx
  __int64 v63; // rdx
  char *v64; // rax
  __int64 v65; // rax
  int v66; // eax
  int v67; // r9d
  _QWORD *v71; // [rsp+18h] [rbp-258h]
  __int64 v72; // [rsp+20h] [rbp-250h]
  __int64 i; // [rsp+28h] [rbp-248h]
  char v74; // [rsp+30h] [rbp-240h]
  __int64 v75; // [rsp+30h] [rbp-240h]
  _QWORD *v76; // [rsp+38h] [rbp-238h]
  __int64 v77; // [rsp+38h] [rbp-238h]
  __int64 *v78; // [rsp+40h] [rbp-230h] BYREF
  __int64 v79; // [rsp+48h] [rbp-228h]
  _BYTE v80[48]; // [rsp+50h] [rbp-220h] BYREF
  __int64 v81; // [rsp+80h] [rbp-1F0h] BYREF
  __int64 *v82; // [rsp+88h] [rbp-1E8h]
  __int64 v83; // [rsp+90h] [rbp-1E0h]
  int v84; // [rsp+98h] [rbp-1D8h]
  char v85; // [rsp+9Ch] [rbp-1D4h]
  char v86; // [rsp+A0h] [rbp-1D0h] BYREF
  __int64 v87; // [rsp+120h] [rbp-150h] BYREF
  char *v88; // [rsp+128h] [rbp-148h]
  __int64 v89; // [rsp+130h] [rbp-140h]
  int v90; // [rsp+138h] [rbp-138h]
  char v91; // [rsp+13Ch] [rbp-134h]
  char v92; // [rsp+140h] [rbp-130h] BYREF

  v6 = a3 - 24;
  if ( !a3 )
    v6 = 0;
  v7 = sub_AA48A0(v6);
  v8 = *(_DWORD *)(a1 + 352);
  v76 = (_QWORD *)v7;
  *(_DWORD *)(a1 + 352) = v8 + 1;
  v9 = unk_3F8E4CC;
  v10 = sub_BD2C40(88, unk_3F8E4CC);
  if ( v10 )
  {
    v9 = sub_BCB120(v76);
    sub_BD35F0((__int64)v10, v9, 27);
    v11 = *((_DWORD *)v10 + 1);
    v10[4] = 0;
    v10[5] = 0;
    v10[6] = 0;
    v12 = *(_BYTE *)v10 == 26;
    v10[7] = 0;
    *((_DWORD *)v10 + 1) = v11 & 0x38000000 | 2;
    v10[8] = v6;
    v10[3] = sub_103AB40;
    v13 = v10 - 8;
    if ( v12 )
      v13 = v10 - 4;
    v10[9] = 0;
    if ( *v13 )
    {
      v14 = v13[1];
      *(_QWORD *)v13[2] = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = v13[2];
    }
    *v13 = 0;
    *((_DWORD *)v10 + 20) = v8;
    *((_DWORD *)v10 + 21) = -1;
  }
  v15 = *(_QWORD *)(a1 + 128);
  *(_QWORD *)(a1 + 128) = v10;
  if ( v15 )
    sub_BD72D0(v15, v9);
  v87 = 0;
  v88 = &v92;
  v89 = 32;
  v90 = 0;
  v91 = 1;
  for ( i = a3; a4 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    v16 = *(_QWORD *)(i + 32);
    v17 = 0;
    v72 = i - 24;
    v18 = 0;
    v77 = i + 24;
    v74 = 0;
    if ( v16 == i + 24 )
      continue;
    do
    {
      while ( 1 )
      {
        v19 = v16 - 24;
        if ( !v16 )
          v19 = 0;
        v20 = sub_10406B0(a1, v19, a2, 0);
        v24 = (__int64)v20;
        if ( v20 )
        {
          if ( !v17 )
          {
            v71 = v20;
            v60 = sub_10416E0(a1, v72);
            v24 = (__int64)v71;
            v17 = (__int64 *)v60;
          }
          v25 = *v17;
          v26 = *(_QWORD *)(v24 + 32);
          *(_QWORD *)(v24 + 40) = v17;
          v25 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v24 + 32) = v25 | v26 & 7;
          *(_QWORD *)(v25 + 8) = v24 + 32;
          v21 = *v17 & 7 | (v24 + 32);
          *v17 = v21;
          if ( *(_BYTE *)v24 == 27 )
            break;
        }
        v16 = *(_QWORD *)(v16 + 8);
        if ( v77 == v16 )
          goto LABEL_26;
      }
      if ( !v18 )
      {
        v75 = v24;
        v65 = sub_1041AC0(a1, v72);
        v24 = v75;
        v18 = (__int64 *)v65;
      }
      v27 = *v18;
      v28 = *(_QWORD *)(v24 + 48);
      *(_QWORD *)(v24 + 56) = v18;
      v74 = 1;
      v27 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v24 + 48) = v27 | v28 & 7;
      *(_QWORD *)(v27 + 8) = v24 + 48;
      v21 = *v18 & 7 | (v24 + 48);
      *v18 = v21;
      v16 = *(_QWORD *)(v16 + 8);
    }
    while ( v77 != v16 );
LABEL_26:
    if ( !v74 )
      continue;
    if ( !v91 )
      goto LABEL_78;
    v64 = v88;
    v21 = HIDWORD(v89);
    v24 = (__int64)&v88[8 * HIDWORD(v89)];
    if ( v88 != (char *)v24 )
    {
      while ( v72 != *(_QWORD *)v64 )
      {
        v64 += 8;
        if ( (char *)v24 == v64 )
          goto LABEL_77;
      }
      continue;
    }
LABEL_77:
    if ( HIDWORD(v89) < (unsigned int)v89 )
    {
      ++HIDWORD(v89);
      *(_QWORD *)v24 = v72;
      ++v87;
    }
    else
    {
LABEL_78:
      sub_C8CC70((__int64)&v87, v72, v24, v21, v22, v23);
    }
  }
  sub_10422E0(a1, (__int64)&v87);
  v29 = *(_QWORD *)(a1 + 24);
  v81 = 0;
  v82 = (__int64 *)&v86;
  v83 = 16;
  v84 = 0;
  v85 = 1;
  if ( !v29 )
  {
    v54 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 96LL);
    result = (_QWORD *)sub_103C0D0(a1, v54, *(__int64 ****)(a1 + 128), (__int64)&v81, 0, 0);
    goto LABEL_47;
  }
  v30 = sub_D4B130(v29);
  v31 = *(_DWORD *)(a1 + 56);
  v32 = *(_QWORD *)(a1 + 40);
  v33 = v30;
  if ( v31 )
  {
    v34 = v31 - 1;
    v35 = v34 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
    v36 = (__int64 *)(v32 + 16LL * v35);
    v37 = *v36;
    if ( v33 != *v36 )
    {
      v66 = 1;
      while ( v37 != -4096 )
      {
        v67 = v66 + 1;
        v35 = v34 & (v66 + v35);
        v36 = (__int64 *)(v32 + 16LL * v35);
        v37 = *v36;
        if ( v33 == *v36 )
          goto LABEL_31;
        v66 = v67;
      }
      goto LABEL_34;
    }
LABEL_31:
    v38 = v36[1];
    if ( v38 )
    {
      v39 = *(_QWORD *)(v38 + 16);
      if ( !v39 )
      {
LABEL_33:
        sub_103CDC0(a1, v38, 1);
        goto LABEL_34;
      }
      while ( 1 )
      {
        v62 = *(_QWORD *)(v39 + 8);
        v63 = *(_QWORD *)(a1 + 128);
        if ( *(_QWORD *)v39 )
        {
          **(_QWORD **)(v39 + 16) = v62;
          if ( v62 )
          {
            *(_QWORD *)(v62 + 16) = *(_QWORD *)(v39 + 16);
            *(_QWORD *)v39 = v63;
            if ( !v63 )
              goto LABEL_67;
          }
          else
          {
            *(_QWORD *)v39 = v63;
            if ( !v63 )
              goto LABEL_33;
          }
        }
        else
        {
          *(_QWORD *)v39 = v63;
          if ( !v63 )
            goto LABEL_66;
        }
        v61 = *(_QWORD *)(v63 + 16);
        *(_QWORD *)(v39 + 8) = v61;
        if ( v61 )
          *(_QWORD *)(v61 + 16) = v39 + 8;
        *(_QWORD *)(v39 + 16) = v63 + 16;
        *(_QWORD *)(v63 + 16) = v39;
LABEL_66:
        if ( !v62 )
          goto LABEL_33;
LABEL_67:
        v39 = v62;
      }
    }
  }
LABEL_34:
  v40 = *(_QWORD *)(a1 + 24);
  v79 = 0x600000000LL;
  v78 = (__int64 *)v80;
  sub_D472F0(v40, (__int64)&v78);
  v45 = v78;
  v46 = v85;
  v47 = &v78[(unsigned int)v79];
  if ( v78 != v47 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v48 = *v45;
        if ( v46 )
          break;
LABEL_81:
        ++v45;
        sub_C8CC70((__int64)&v81, v48, (__int64)v41, v42, v43, v44);
        v46 = v85;
        if ( v47 == v45 )
          goto LABEL_41;
      }
      v49 = v82;
      v42 = HIDWORD(v83);
      v41 = &v82[HIDWORD(v83)];
      if ( v82 == v41 )
      {
LABEL_87:
        if ( HIDWORD(v83) >= (unsigned int)v83 )
          goto LABEL_81;
        v42 = (unsigned int)(HIDWORD(v83) + 1);
        ++v45;
        ++HIDWORD(v83);
        *v41 = v48;
        v46 = v85;
        ++v81;
        if ( v47 == v45 )
          break;
      }
      else
      {
        while ( v48 != *v49 )
        {
          if ( v41 == ++v49 )
            goto LABEL_87;
        }
        if ( v47 == ++v45 )
          break;
      }
    }
  }
LABEL_41:
  v50 = *(__int64 ****)(a1 + 128);
  v51 = *(_QWORD *)(a1 + 8);
  v52 = sub_D4B130(*(_QWORD *)(a1 + 24));
  v53 = 0;
  if ( v52 )
  {
    v53 = (unsigned int)(*(_DWORD *)(v52 + 44) + 1);
    LODWORD(v52) = *(_DWORD *)(v52 + 44) + 1;
  }
  v54 = 0;
  if ( (unsigned int)v52 < *(_DWORD *)(v51 + 32) )
    v54 = *(__int64 **)(*(_QWORD *)(v51 + 24) + 8 * v53);
  result = (_QWORD *)sub_103C0D0(a1, v54, v50, (__int64)&v81, 0, 0);
  if ( v78 != (__int64 *)v80 )
    result = (_QWORD *)_libc_free(v78, v54);
LABEL_47:
  v56 = v85;
  if ( a3 != a4 )
  {
    v57 = a3;
    do
    {
      while ( 1 )
      {
        v58 = v57 - 24;
        if ( !v57 )
          v58 = 0;
        if ( !v56 )
          break;
        result = v82;
        v59 = &v82[HIDWORD(v83)];
        if ( v82 == v59 )
          goto LABEL_84;
        while ( v58 != *result )
        {
          if ( v59 == ++result )
            goto LABEL_84;
        }
        v57 = *(_QWORD *)(v57 + 8);
        if ( a4 == v57 )
          goto LABEL_57;
      }
      v54 = (__int64 *)v58;
      result = sub_C8CA60((__int64)&v81, v58);
      if ( !result )
      {
LABEL_84:
        v54 = (__int64 *)v58;
        result = (_QWORD *)sub_103C5C0(a1, v58);
      }
      v57 = *(_QWORD *)(v57 + 8);
      v56 = v85;
    }
    while ( a4 != v57 );
  }
LABEL_57:
  if ( !v56 )
    result = (_QWORD *)_libc_free(v82, v54);
  if ( !v91 )
    return (_QWORD *)_libc_free(v88, v54);
  return result;
}
