// Function: sub_1DC0580
// Address: 0x1dc0580
//
__int64 __fastcall sub_1DC0580(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  char v9; // r14
  int v10; // eax
  int v11; // r14d
  __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rdi
  __int64 v16; // rcx
  unsigned __int64 i; // rdx
  __int64 v18; // rsi
  __int64 v19; // rcx
  unsigned int v20; // edi
  __int64 *v21; // rax
  __int64 v22; // r9
  __int64 v23; // r13
  __int64 *v24; // rcx
  __int64 v25; // rsi
  unsigned int v26; // edi
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // rax
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r8
  __int64 *v34; // r12
  __int64 *v35; // rbx
  __int64 v36; // rdx
  __int64 v37; // rcx
  int v38; // r8d
  int v39; // r9d
  unsigned int v40; // eax
  __int64 v41; // r14
  unsigned int v42; // r13d
  __int64 v43; // r12
  __int64 v44; // rdi
  int v46; // eax
  int v47; // r10d
  __int64 v48; // rax
  __int128 v49; // [rsp-20h] [rbp-220h]
  __int64 v51; // [rsp+20h] [rbp-1E0h]
  unsigned __int64 v52; // [rsp+28h] [rbp-1D8h]
  __int64 v53; // [rsp+28h] [rbp-1D8h]
  unsigned __int64 v54[2]; // [rsp+50h] [rbp-1B0h] BYREF
  _BYTE v55[48]; // [rsp+60h] [rbp-1A0h] BYREF
  _BYTE *v56; // [rsp+90h] [rbp-170h]
  __int64 v57; // [rsp+98h] [rbp-168h]
  _BYTE v58[16]; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v59; // [rsp+B0h] [rbp-150h]
  _BYTE *v60; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v61; // [rsp+C8h] [rbp-138h]
  _BYTE v62[304]; // [rsp+D0h] [rbp-130h] BYREF

  v7 = (__int64)a1;
  v8 = *(_QWORD *)(a2 + 104);
  if ( v8 )
  {
    v9 = 0;
    do
    {
      sub_1DBFFB0(a1, v8, *(_DWORD *)(a2 + 112), a4, a5, a6);
      v10 = *(_DWORD *)(v8 + 8);
      v8 = *(_QWORD *)(v8 + 104);
      if ( !v10 )
        v9 = 1;
    }
    while ( v8 );
    if ( v9 )
      sub_1DB4C70(a2);
  }
  v11 = *(_DWORD *)(a2 + 112);
  v12 = a1[30];
  v60 = v62;
  v61 = 0x1000000000LL;
  if ( v11 < 0 )
  {
    v13 = *(_QWORD *)(*(_QWORD *)(v12 + 24) + 16LL * (v11 & 0x7FFFFFFF) + 8);
    if ( !v13 )
      goto LABEL_31;
  }
  else
  {
    v13 = *(_QWORD *)(*(_QWORD *)(v12 + 272) + 8LL * (unsigned int)v11);
    if ( !v13 )
      goto LABEL_31;
  }
  v14 = *(_QWORD *)(v13 + 16);
LABEL_11:
  v15 = v14;
  if ( **(_WORD **)(v14 + 16) != 12 )
  {
    if ( !(unsigned __int8)sub_1E166B0(v14, (unsigned int)v11, 0) )
      goto LABEL_28;
    v16 = *(_QWORD *)(v7 + 272);
    for ( i = v14; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
      ;
    v18 = *(_QWORD *)(v16 + 368);
    v19 = *(unsigned int *)(v16 + 384);
    if ( (_DWORD)v19 )
    {
      v20 = (v19 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
      v21 = (__int64 *)(v18 + 16LL * v20);
      v22 = *v21;
      if ( i == *v21 )
      {
LABEL_17:
        v52 = v21[1] & 0xFFFFFFFFFFFFFFF8LL;
        v23 = v52 | 4;
        v24 = (__int64 *)sub_1DB3C70((__int64 *)a2, v52);
        v25 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
        if ( v24 == (__int64 *)v25 )
          goto LABEL_28;
        a6 = *(unsigned int *)(v52 + 24);
        v26 = *(_DWORD *)((*v24 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        if ( (unsigned __int64)(v26 | (*v24 >> 1) & 3) > (unsigned int)a6 )
          goto LABEL_28;
        v27 = v24[2];
        if ( v52 == (v24[1] & 0xFFFFFFFFFFFFFFF8LL) )
        {
          if ( (__int64 *)v25 == v24 + 3 )
          {
            if ( !v27 )
              goto LABEL_28;
            goto LABEL_25;
          }
          v26 = *(_DWORD *)((v24[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
          v24 += 3;
        }
        if ( v52 == *(_QWORD *)(v27 + 8) )
          goto LABEL_28;
        if ( v26 <= (unsigned int)a6 )
        {
          v28 = v24[2];
          if ( v27 != v28 )
          {
            if ( v28 )
              v23 = *(_QWORD *)(v28 + 8);
          }
        }
LABEL_25:
        v29 = (unsigned int)v61;
        if ( (unsigned int)v61 >= HIDWORD(v61) )
        {
          v53 = v27;
          sub_16CD150((__int64)&v60, v62, 0, 16, v27, a6);
          v29 = (unsigned int)v61;
          v27 = v53;
        }
        v30 = (__int64 *)&v60[16 * v29];
        *v30 = v23;
        v30[1] = v27;
        LODWORD(v61) = v61 + 1;
LABEL_28:
        v15 = *(_QWORD *)(v13 + 16);
        v13 = *(_QWORD *)(v13 + 32);
        if ( v13 )
          goto LABEL_29;
        goto LABEL_31;
      }
      v46 = 1;
      while ( v22 != -8 )
      {
        v47 = v46 + 1;
        v48 = ((_DWORD)v19 - 1) & (v20 + v46);
        v20 = v48;
        v21 = (__int64 *)(v18 + 16 * v48);
        v22 = *v21;
        if ( *v21 == i )
          goto LABEL_17;
        v46 = v47;
      }
    }
    v21 = (__int64 *)(v18 + 16 * v19);
    goto LABEL_17;
  }
  while ( 1 )
  {
    v13 = *(_QWORD *)(v13 + 32);
    if ( !v13 )
      break;
LABEL_29:
    v14 = *(_QWORD *)(v13 + 16);
    if ( v14 != v15 )
      goto LABEL_11;
  }
LABEL_31:
  v31 = *(__int64 **)(a2 + 64);
  v54[1] = 0x200000000LL;
  v57 = 0x200000000LL;
  v32 = *(unsigned int *)(a2 + 72);
  v54[0] = (unsigned __int64)v55;
  v33 = (__int64)&v31[v32];
  v56 = v58;
  v59 = 0;
  if ( v31 != (__int64 *)v33 )
  {
    v51 = v7;
    v34 = &v31[v32];
    v35 = v31;
    do
    {
      if ( (*(_QWORD *)(*v35 + 8) & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        *((_QWORD *)&v49 + 1) = *(_QWORD *)(*v35 + 8) & 0xFFFFFFFFFFFFFFF8LL | 6;
        *(_QWORD *)&v49 = *(_QWORD *)(*v35 + 8);
        sub_1DB8610((__int64)v54, (__int64)v58, *v35, *((__int64 *)&v49 + 1), v33, a6, v49, *v35);
      }
      ++v35;
    }
    while ( v34 != v35 );
    v7 = v51;
  }
  sub_1DBB5C0(v7, (__int64)v54, (__int64)&v60, v11, 0, a6);
  sub_1DBFD90(a2, (__int64)v54, v36, v37, v38, v39);
  v40 = sub_1DBAEC0(v7, a2, a3);
  v41 = v59;
  v42 = v40;
  if ( v59 )
  {
    v43 = *(_QWORD *)(v59 + 16);
    while ( v43 )
    {
      sub_1DB97B0(*(_QWORD *)(v43 + 24));
      v44 = v43;
      v43 = *(_QWORD *)(v43 + 16);
      j_j___libc_free_0(v44, 56);
    }
    j_j___libc_free_0(v41, 48);
  }
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
  if ( (_BYTE *)v54[0] != v55 )
    _libc_free(v54[0]);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  return v42;
}
