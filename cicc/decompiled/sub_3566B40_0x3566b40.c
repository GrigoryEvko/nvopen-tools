// Function: sub_3566B40
// Address: 0x3566b40
//
__int64 __fastcall sub_3566B40(_QWORD *a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // r13d
  _QWORD v18[8]; // [rsp+0h] [rbp-19A0h] BYREF
  _QWORD v19[9]; // [rsp+40h] [rbp-1960h] BYREF
  __int64 v20; // [rsp+88h] [rbp-1918h]
  char *v21; // [rsp+90h] [rbp-1910h]
  char v22; // [rsp+A0h] [rbp-1900h] BYREF
  char *v23; // [rsp+D0h] [rbp-18D0h]
  char v24; // [rsp+E0h] [rbp-18C0h] BYREF
  __int64 v25; // [rsp+118h] [rbp-1888h]
  unsigned int v26; // [rsp+128h] [rbp-1878h]
  __int64 v27; // [rsp+138h] [rbp-1868h]
  unsigned int v28; // [rsp+148h] [rbp-1858h]
  char *v29; // [rsp+150h] [rbp-1850h]
  char v30; // [rsp+160h] [rbp-1840h] BYREF

  sub_2EC5460(v18);
  v3 = (__int64 *)a1[1];
  v18[1] = a1[25];
  v18[2] = a1[27];
  v18[3] = a1[28];
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_29:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_5027190 )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_29;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_5027190);
  v7 = (__int64 *)a1[1];
  v18[4] = *(_QWORD *)(v6 + 256);
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_27:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4F86530 )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_27;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_4F86530);
  v11 = (__int64 *)a1[1];
  v18[5] = *(_QWORD *)(v10 + 176);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_28:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_501EACC )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_28;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_501EACC);
  v15 = a1[25];
  v18[6] = v14 + 200;
  sub_2F5FFA0((_QWORD *)v18[7], v15);
  sub_35E6C40(v19, v18, a2);
  v16 = sub_35ECD10(v19);
  v19[0] = &unk_4A3ADC8;
  if ( v29 != &v30 )
    _libc_free((unsigned __int64)v29);
  sub_C7D6A0(v27, 16LL * v28, 8);
  sub_C7D6A0(v25, 16LL * v26, 8);
  if ( v23 != &v24 )
    _libc_free((unsigned __int64)v23);
  if ( v21 != &v22 )
    _libc_free((unsigned __int64)v21);
  if ( v20 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v20 + 8LL))(v20);
  sub_2EC14D0(v18);
  return v16;
}
