// Function: sub_234CFF0
// Address: 0x234cff0
//
__int64 __fastcall sub_234CFF0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v14; // [rsp+0h] [rbp-50h]
  __int64 v15; // [rsp+8h] [rbp-48h] BYREF
  __int64 v16; // [rsp+10h] [rbp-40h]
  __int64 v17; // [rsp+18h] [rbp-38h]
  __int64 v18; // [rsp+20h] [rbp-30h]
  __int64 v19; // [rsp+28h] [rbp-28h]
  int v20; // [rsp+30h] [rbp-20h]

  v4 = *a2;
  *a2 = 0;
  v18 = 0;
  v14 = v4;
  v5 = a2[1];
  a2[1] = 0;
  v15 = v5;
  v6 = a2[2];
  a2[2] = 0;
  v16 = v6;
  v7 = a2[3];
  a2[3] = 0;
  v17 = v7;
  LODWORD(v7) = *((_DWORD *)a2 + 12);
  v19 = 0;
  v20 = v7;
  v8 = sub_22077B0(0x40u);
  if ( v8 )
  {
    *(_QWORD *)(v8 + 40) = 0;
    *(_QWORD *)(v8 + 48) = 0;
    *(_QWORD *)v8 = &unk_4A11E38;
    v9 = v14;
    v14 = 0;
    *(_QWORD *)(v8 + 8) = v9;
    v10 = v15;
    v15 = 0;
    *(_QWORD *)(v8 + 16) = v10;
    v11 = v16;
    v16 = 0;
    *(_QWORD *)(v8 + 24) = v11;
    v12 = v17;
    v17 = 0;
    *(_QWORD *)(v8 + 32) = v12;
    *(_DWORD *)(v8 + 56) = v20;
  }
  *(_QWORD *)a1 = v8;
  *(_BYTE *)(a1 + 8) = a3;
  sub_233F7F0((__int64)&v15);
  if ( v14 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  return a1;
}
