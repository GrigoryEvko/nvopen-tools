// Function: sub_228DB90
// Address: 0x228db90
//
__int64 __fastcall sub_228DB90(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, _QWORD *a5, __int64 *a6)
{
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12
  unsigned __int64 v13; // r13
  char v14; // al
  unsigned __int64 v15; // rdi
  unsigned int v16; // eax
  unsigned __int64 v17; // r14
  unsigned int v18; // r12d
  char v19; // si
  unsigned __int64 *v20; // r13
  char v21; // r15
  __int64 *v23; // rbx
  __int64 v24; // r13
  __int64 v25; // rdi
  __int64 v26; // r14
  int v27; // eax
  _QWORD *v28; // rax
  _QWORD *v29; // r13
  __int64 *v30; // rbx
  __int64 v31; // rdi
  __int64 v32; // r15
  __int64 *v33; // rbx
  __int64 v34; // r13
  __int64 v35; // rdi
  __int64 v37; // [rsp+8h] [rbp-58h]
  unsigned int v39; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41; // [rsp+20h] [rbp-40h] BYREF
  __int64 v42[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_B48880((__int64 *)&v41, *(_DWORD *)(a1 + 40) + 1, 0);
  sub_B48880(v42, *(_DWORD *)(a1 + 40) + 1, 0);
  if ( !sub_228DB70(a1, a2, a3, (__int64)&v41) || !sub_228DB80(a1, a4, a5, (__int64)v42) )
  {
    v17 = v42[0];
    v18 = 4;
    v19 = v42[0] & 1;
    goto LABEL_19;
  }
  v12 = v41;
  v13 = *a6;
  v14 = v41 & 1;
  if ( (*a6 & 1) != 0 )
  {
    if ( v14 )
    {
LABEL_5:
      *a6 = v12;
      goto LABEL_6;
    }
    v28 = (_QWORD *)sub_22077B0(0x48u);
    v29 = v28;
    if ( v28 )
    {
      *v28 = v28 + 2;
      v28[1] = 0x600000000LL;
      if ( *(_DWORD *)(v12 + 8) )
        sub_228AB60((__int64)v28, v12, v8, v9, v10, v11);
      *((_DWORD *)v29 + 16) = *(_DWORD *)(v12 + 64);
    }
    *a6 = (__int64)v29;
  }
  else
  {
    if ( v14 )
    {
      if ( v13 )
      {
        if ( *(_QWORD *)v13 != v13 + 16 )
          _libc_free(*(_QWORD *)v13);
        j_j___libc_free_0(v13);
        v12 = v41;
      }
      goto LABEL_5;
    }
    sub_228AB60(*a6, v41, v8, v9, v10, v11);
    *(_DWORD *)(v13 + 64) = *(_DWORD *)(v12 + 64);
  }
LABEL_6:
  sub_228C270((unsigned __int64 *)a6, (unsigned __int64 *)v42, v8, v9, v10, v11);
  v15 = *a6;
  if ( (*a6 & 1) != 0 )
  {
    v16 = sub_39FAC40(~(-1LL << (v15 >> 58)) & (v15 >> 1));
    v17 = v42[0];
    v18 = v16;
    v19 = v42[0] & 1;
    if ( v16 )
      goto LABEL_8;
LABEL_27:
    v18 = 0;
    goto LABEL_19;
  }
  v23 = *(__int64 **)v15;
  v18 = 0;
  v24 = *(_QWORD *)v15 + 8LL * *(unsigned int *)(v15 + 8);
  if ( *(_QWORD *)v15 == v24 )
  {
    v17 = v42[0];
    v19 = v42[0] & 1;
    goto LABEL_19;
  }
  do
  {
    v25 = *v23++;
    v18 += sub_39FAC40(v25);
  }
  while ( (__int64 *)v24 != v23 );
  v17 = v42[0];
  v19 = v42[0] & 1;
  if ( !v18 )
    goto LABEL_27;
LABEL_8:
  if ( v18 == 1 )
    goto LABEL_19;
  if ( v18 != 2 )
  {
    v18 = 3;
    if ( v19 )
      goto LABEL_11;
LABEL_20:
    if ( v17 )
    {
      if ( *(_QWORD *)v17 != v17 + 16 )
        _libc_free(*(_QWORD *)v17);
      j_j___libc_free_0(v17);
    }
    goto LABEL_11;
  }
  v20 = (unsigned __int64 *)v41;
  v21 = v41 & 1;
  if ( (v41 & 1) != 0 )
  {
    v40 = (int)sub_39FAC40((v41 >> 1) & ~(-1LL << (v41 >> 58)));
    goto LABEL_35;
  }
  v37 = *(_QWORD *)v41 + 8LL * *(unsigned int *)(v41 + 8);
  if ( *(_QWORD *)v41 == v37 )
  {
LABEL_19:
    if ( v19 )
      goto LABEL_11;
    goto LABEL_20;
  }
  v39 = 0;
  v30 = *(__int64 **)v41;
  do
  {
    v31 = *v30++;
    v39 += sub_39FAC40(v31);
  }
  while ( (__int64 *)v37 != v30 );
  v40 = v39;
LABEL_35:
  if ( !v40 )
    goto LABEL_19;
  if ( !v19 )
  {
    v32 = *(_QWORD *)v17 + 8LL * *(unsigned int *)(v17 + 8);
    if ( *(_QWORD *)v17 != v32 )
    {
      v33 = *(__int64 **)v17;
      LODWORD(v34) = 0;
      do
      {
        v35 = *v33++;
        v34 = (unsigned int)sub_39FAC40(v35) + (unsigned int)v34;
      }
      while ( (__int64 *)v32 != v33 );
      if ( v34 && (v34 != 1 || v40 != 1) )
        v18 = 3;
    }
    goto LABEL_20;
  }
  v26 = ~(-1LL << (v17 >> 58)) & (v17 >> 1);
  v27 = sub_39FAC40(v26);
  if ( !v26 )
    goto LABEL_12;
  if ( v40 != 1 || v27 != 1 )
    v18 = 3;
LABEL_11:
  v20 = (unsigned __int64 *)v41;
  v21 = v41 & 1;
LABEL_12:
  if ( !v21 && v20 )
  {
    if ( (unsigned __int64 *)*v20 != v20 + 2 )
      _libc_free(*v20);
    j_j___libc_free_0((unsigned __int64)v20);
  }
  return v18;
}
