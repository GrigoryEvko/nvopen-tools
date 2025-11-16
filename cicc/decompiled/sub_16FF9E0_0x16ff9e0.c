// Function: sub_16FF9E0
// Address: 0x16ff9e0
//
void __fastcall sub_16FF9E0(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // r12
  __int64 v4; // rdi
  _QWORD *v5; // r12
  __int64 (__fastcall *v6)(_QWORD *); // rax
  _QWORD *v7; // rdi
  _QWORD *v8; // rdi
  _QWORD *v9; // r13
  _QWORD *v10; // r12
  _QWORD *v11; // rdi
  _QWORD *v12; // rdi
  _QWORD *v13; // r13
  _QWORD *v14; // r12
  _QWORD *v15; // rdi
  _QWORD *v16; // rdi
  _QWORD *v17; // rdi
  _QWORD *v18; // rdi
  _QWORD *v19; // rdi

  *a1 = &unk_49EFE98;
  v2 = a1[76];
  if ( v2 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  v3 = a1[77];
  if ( v3 )
  {
    j___libc_free_0(*(_QWORD *)(v3 + 200));
    j___libc_free_0(*(_QWORD *)(v3 + 168));
    j_j___libc_free_0(v3, 224);
  }
  v4 = a1[78];
  if ( v4 )
    j_j___libc_free_0(v4, 32);
  v5 = (_QWORD *)a1[79];
  if ( v5 )
  {
    v6 = *(__int64 (__fastcall **)(_QWORD *))(*v5 + 8LL);
    if ( v6 == sub_168C4D0 )
    {
      v7 = (_QWORD *)v5[8];
      *v5 = &unk_49EE580;
      if ( v7 != v5 + 10 )
        j_j___libc_free_0(v7, v5[10] + 1LL);
      v8 = (_QWORD *)v5[1];
      if ( v8 != v5 + 3 )
        j_j___libc_free_0(v8, v5[3] + 1LL);
      j_j___libc_free_0(v5, 216);
    }
    else
    {
      v6((_QWORD *)a1[79]);
    }
  }
  v9 = (_QWORD *)a1[115];
  v10 = (_QWORD *)a1[114];
  if ( v9 != v10 )
  {
    do
    {
      if ( (_QWORD *)*v10 != v10 + 2 )
        j_j___libc_free_0(*v10, v10[2] + 1LL);
      v10 += 4;
    }
    while ( v9 != v10 );
    v10 = (_QWORD *)a1[114];
  }
  if ( v10 )
    j_j___libc_free_0(v10, a1[116] - (_QWORD)v10);
  v11 = (_QWORD *)a1[110];
  if ( v11 != a1 + 112 )
    j_j___libc_free_0(v11, a1[112] + 1LL);
  v12 = (_QWORD *)a1[106];
  if ( v12 != a1 + 108 )
    j_j___libc_free_0(v12, a1[108] + 1LL);
  v13 = (_QWORD *)a1[97];
  v14 = (_QWORD *)a1[96];
  if ( v13 != v14 )
  {
    do
    {
      if ( (_QWORD *)*v14 != v14 + 2 )
        j_j___libc_free_0(*v14, v14[2] + 1LL);
      v14 += 4;
    }
    while ( v13 != v14 );
    v14 = (_QWORD *)a1[96];
  }
  if ( v14 )
    j_j___libc_free_0(v14, a1[98] - (_QWORD)v14);
  v15 = (_QWORD *)a1[92];
  if ( v15 != a1 + 94 )
    j_j___libc_free_0(v15, a1[94] + 1LL);
  v16 = (_QWORD *)a1[88];
  if ( v16 != a1 + 90 )
    j_j___libc_free_0(v16, a1[90] + 1LL);
  v17 = (_QWORD *)a1[70];
  if ( v17 != a1 + 72 )
    j_j___libc_free_0(v17, a1[72] + 1LL);
  v18 = (_QWORD *)a1[66];
  if ( v18 != a1 + 68 )
    j_j___libc_free_0(v18, a1[68] + 1LL);
  v19 = (_QWORD *)a1[59];
  if ( v19 != a1 + 61 )
    j_j___libc_free_0(v19, a1[61] + 1LL);
  sub_15A93E0(a1 + 2);
}
