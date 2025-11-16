// Function: sub_234DD80
// Address: 0x234dd80
//
__int64 __fastcall sub_234DD80(__int64 a1, __int64 *a2, char a3, char a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v16; // [rsp+0h] [rbp-60h]
  __int64 v17; // [rsp+8h] [rbp-58h] BYREF
  __int64 v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h]
  __int64 v21; // [rsp+28h] [rbp-38h]
  int v22; // [rsp+30h] [rbp-30h]

  v6 = *a2;
  *a2 = 0;
  v20 = 0;
  v16 = v6;
  v7 = a2[1];
  a2[1] = 0;
  v17 = v7;
  v8 = a2[2];
  a2[2] = 0;
  v18 = v8;
  v9 = a2[3];
  a2[3] = 0;
  v19 = v9;
  LODWORD(v9) = *((_DWORD *)a2 + 12);
  v21 = 0;
  v22 = v9;
  v10 = sub_22077B0(0x40u);
  if ( v10 )
  {
    *(_QWORD *)(v10 + 40) = 0;
    *(_QWORD *)(v10 + 48) = 0;
    *(_QWORD *)v10 = &unk_4A11E38;
    v11 = v16;
    v16 = 0;
    *(_QWORD *)(v10 + 8) = v11;
    v12 = v17;
    v17 = 0;
    *(_QWORD *)(v10 + 16) = v12;
    v13 = v18;
    v18 = 0;
    *(_QWORD *)(v10 + 24) = v13;
    v14 = v19;
    v19 = 0;
    *(_QWORD *)(v10 + 32) = v14;
    *(_DWORD *)(v10 + 56) = v22;
  }
  *(_QWORD *)a1 = v10;
  *(_BYTE *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 9) = a4;
  sub_233F7F0((__int64)&v17);
  if ( v16 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
  return a1;
}
