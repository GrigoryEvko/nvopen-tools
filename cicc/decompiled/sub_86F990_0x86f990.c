// Function: sub_86F990
// Address: 0x86f990
//
_BYTE *__fastcall sub_86F990(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r13
  _BYTE *v8; // rax
  _QWORD *v9; // r15
  _QWORD *v10; // rax
  _BYTE *v11; // r12
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 i; // rax
  __int64 v16; // rax
  char v17; // si
  char v18; // al
  _QWORD *v20; // r13
  _BYTE *v21; // rax
  _QWORD *v22; // r15
  __int64 v23; // rax
  _BYTE *v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 *v30; // r9
  _BYTE *v31; // rax
  __int64 *v32; // [rsp+8h] [rbp-48h]
  _QWORD *v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  _QWORD *v35; // [rsp+18h] [rbp-38h]
  _BYTE *v36; // [rsp+18h] [rbp-38h]

  v5 = a2;
  v34 = sub_72B840(a1);
  v8 = sub_726B30(11);
  a2[3] = v8;
  v9 = v8;
  if ( a4 )
  {
    v10 = sub_726B30(0);
    v10[6] = a4;
    v10[3] = v9;
    v10[2] = a2;
    v5 = v10;
  }
  v9[9] = v5;
  v11 = v9;
  if ( dword_4D048B8 )
  {
    v11 = sub_726B30(19);
    *(_QWORD *)(*((_QWORD *)v11 + 9) + 8LL) = v9;
    v9[3] = v11;
    sub_733780(0x13u, *((_QWORD *)v11 + 9), 0, 5, 0);
  }
  sub_733780(0x14u, v9[10], 0, 1, 0);
  v12 = *(_QWORD *)(v34 + 88);
  v13 = *(_QWORD *)(v12 + 48);
  v14 = *(_QWORD *)(v13 + 56);
  if ( dword_4D048B8 )
    v13 = *(_QWORD *)(v13 + 48);
  for ( i = v14; i; i = *(_QWORD *)(i + 56) )
    *(_QWORD *)(i + 32) = v13;
  *(_QWORD *)(v13 + 48) = v14;
  *(_QWORD *)(*(_QWORD *)(v12 + 48) + 56LL) = 0;
  *(_QWORD *)(*(_QWORD *)(v12 + 48) + 40LL) = 0;
  v16 = *(_QWORD *)(v12 + 24);
  if ( v16 )
  {
    do
    {
      *(_QWORD *)(v16 + 24) = v13;
      v16 = *(_QWORD *)(v16 + 32);
    }
    while ( v16 );
    v16 = *(_QWORD *)(v12 + 24);
  }
  *(_QWORD *)(v13 + 24) = v16;
  v17 = *(_BYTE *)(v12 + 1);
  *(_QWORD *)(v12 + 24) = 0;
  *(_BYTE *)(v13 + 1) = v17 & 1 | *(_BYTE *)(v13 + 1) & 0xFE;
  v18 = *(_BYTE *)(v12 + 1) & 0xFE;
  *(_BYTE *)(v12 + 1) = v18;
  *(_BYTE *)(v13 + 1) = *(_BYTE *)(v13 + 1) & 0xFD | v18 & 2;
  *(_BYTE *)(v12 + 1) &= ~2u;
  sub_733F40();
  if ( dword_4D048B8 )
  {
    sub_8600D0(2u, -1, 0, 0);
    v32 = sub_726810();
    *(_QWORD *)(*((_QWORD *)v11 + 9) + 16LL) = v32;
    sub_7335B0((__int64)v32);
    v20 = sub_726B30(11);
    *(_QWORD *)(v20[10] + 8LL) = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 184);
    v21 = sub_726B30(1);
    v20[9] = v21;
    v22 = v21;
    *((_QWORD *)v21 + 3) = v20;
    *((_QWORD *)v21 + 6) = sub_726700(1);
    v35 = sub_73E830(*(_QWORD *)(a3 + 24));
    v23 = sub_6EFF80();
    sub_73D8E0(v22[6], 0x1Du, v23, 0, (__int64)v35);
    v24 = sub_726B30(0);
    v25 = 8;
    v22[9] = v24;
    v36 = v24;
    v33 = sub_726700(8);
    *((_QWORD *)v36 + 6) = v33;
    v26 = sub_72CBE0();
    v28 = (__int64)v36;
    v29 = (__int64)v33;
    *v33 = v26;
    *(_QWORD *)(*((_QWORD *)v36 + 6) + 56LL) = 0;
    *(_BYTE *)(*((_QWORD *)v36 + 6) + 25LL) |= 4u;
    v30 = v32;
    if ( *(_QWORD *)(a3 + 72) )
    {
      v25 = 0;
      v31 = sub_726B30(0);
      v30 = v32;
      v22[2] = v31;
      v28 = *(_QWORD *)(a3 + 72);
      *((_QWORD *)v31 + 3) = v20;
      *((_QWORD *)v31 + 6) = v28;
    }
    v30[3] = (__int64)v20;
    v20[3] = v11;
    sub_863FC0(v25, 29, v28, v29, v27, v30);
    *(_BYTE *)(a1 + 202) |= 0x10u;
    sub_733F40();
  }
  return v11;
}
