// Function: sub_135DCD0
// Address: 0x135dcd0
//
__int64 __fastcall sub_135DCD0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r13
  _QWORD *v15; // r15
  __int64 v16; // r14
  _QWORD *v17; // rax
  _QWORD *v18; // rdx
  _QWORD *v19; // rcx
  _QWORD *v20; // rax
  __int64 v21; // r12
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+18h] [rbp-38h]
  __int64 v30; // [rsp+18h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_45:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F9D764 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_45;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F9D764);
  v6 = *(__int64 **)(a1 + 8);
  v29 = v5;
  v7 = *v6;
  v8 = v6[1];
  if ( v7 == v8 )
LABEL_46:
    BUG();
  while ( *(_UNKNOWN **)v7 != &unk_4F9B6E8 )
  {
    v7 += 16;
    if ( v8 == v7 )
      goto LABEL_46;
  }
  v9 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v7 + 8) + 104LL))(*(_QWORD *)(v7 + 8), &unk_4F9B6E8);
  v10 = *(__int64 **)(a1 + 8);
  v11 = v9;
  v12 = *v10;
  v13 = v10[1];
  if ( v12 == v13 )
LABEL_47:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F9E06C )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_47;
  }
  v27 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9E06C);
  v14 = sub_160F9A0(*(_QWORD *)(a1 + 8), &unk_4F9920C, 1);
  v26 = v11 + 360;
  if ( !v14 )
  {
    v25 = sub_160F9A0(*(_QWORD *)(a1 + 8), &unk_4F99CC4, 1);
    v15 = (_QWORD *)v25;
    if ( !v25 )
    {
      v28 = v27 + 160;
      v16 = sub_1632FA0(*(_QWORD *)(a2 + 40));
      v30 = sub_14CF090(v29, a2);
      goto LABEL_19;
    }
    v15 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v25 + 104LL))(v25, &unk_4F99CC4);
    v16 = sub_1632FA0(*(_QWORD *)(a2 + 40));
    v30 = sub_14CF090(v29, a2);
    v28 = v27 + 160;
    goto LABEL_17;
  }
  v14 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v14 + 104LL))(v14, &unk_4F9920C);
  v15 = (_QWORD *)sub_160F9A0(*(_QWORD *)(a1 + 8), &unk_4F99CC4, 1);
  v28 = v27 + 160;
  if ( v15 )
  {
    v15 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *, void *))(*v15 + 104LL))(v15, &unk_4F99CC4);
    v16 = sub_1632FA0(*(_QWORD *)(a2 + 40));
    v30 = sub_14CF090(v29, a2);
    if ( v14 )
      v14 += 160;
LABEL_17:
    if ( v15 )
      v15 = (_QWORD *)v15[20];
    goto LABEL_19;
  }
  v16 = sub_1632FA0(*(_QWORD *)(a2 + 40));
  v30 = sub_14CF090(v29, a2);
  if ( v14 )
    v14 += 160;
LABEL_19:
  v17 = (_QWORD *)sub_22077B0(1056);
  v18 = v17;
  if ( v17 )
  {
    v17[1] = v16;
    v19 = v17 + 98;
    v17[2] = a2;
    v17[6] = v14;
    v17[3] = v26;
    v17[7] = v15;
    v17[4] = v30;
    v17[8] = 0;
    v17[9] = 1;
    v17[5] = v28;
    v20 = v17 + 10;
    do
    {
      if ( v20 )
      {
        *v20 = -8;
        v20[1] = 0;
        v20[2] = 0;
        v20[3] = 0;
        v20[4] = 0;
        v20[5] = -8;
        v20[6] = 0;
        v20[7] = 0;
        v20[8] = 0;
        v20[9] = 0;
      }
      v20 += 11;
    }
    while ( v20 != v19 );
    v18[98] = 0;
    v18[99] = v18 + 103;
    v18[100] = v18 + 103;
    v18[101] = 8;
    *((_DWORD *)v18 + 204) = 0;
    v18[111] = 0;
    v18[112] = v18 + 116;
    v18[113] = v18 + 116;
    v18[114] = 16;
    *((_DWORD *)v18 + 230) = 0;
  }
  v21 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v18;
  if ( v21 )
  {
    v22 = *(_QWORD *)(v21 + 904);
    if ( v22 != *(_QWORD *)(v21 + 896) )
      _libc_free(v22);
    v23 = *(_QWORD *)(v21 + 800);
    if ( v23 != *(_QWORD *)(v21 + 792) )
      _libc_free(v23);
    if ( (*(_BYTE *)(v21 + 72) & 1) == 0 )
      j___libc_free_0(*(_QWORD *)(v21 + 80));
    j_j___libc_free_0(v21, 1056);
  }
  return 0;
}
