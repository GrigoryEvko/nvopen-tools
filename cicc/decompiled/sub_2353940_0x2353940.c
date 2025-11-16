// Function: sub_2353940
// Address: 0x2353940
//
void __fastcall sub_2353940(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // [rsp+8h] [rbp-58h] BYREF
  __int64 v12; // [rsp+10h] [rbp-50h]
  __int64 v13; // [rsp+18h] [rbp-48h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h]
  __int64 v15; // [rsp+28h] [rbp-38h]
  __int64 v16; // [rsp+30h] [rbp-30h]
  __int64 v17; // [rsp+38h] [rbp-28h]
  int v18; // [rsp+40h] [rbp-20h]

  v2 = *a2;
  *a2 = 0;
  v16 = 0;
  v12 = v2;
  v3 = a2[1];
  a2[1] = 0;
  v13 = v3;
  v4 = a2[2];
  a2[2] = 0;
  v14 = v4;
  v5 = a2[3];
  a2[3] = 0;
  v15 = v5;
  LODWORD(v5) = *((_DWORD *)a2 + 12);
  v17 = 0;
  v18 = v5;
  v6 = sub_22077B0(0x40u);
  if ( v6 )
  {
    *(_QWORD *)(v6 + 40) = 0;
    *(_QWORD *)(v6 + 48) = 0;
    *(_QWORD *)v6 = &unk_4A11E38;
    v7 = v12;
    v12 = 0;
    *(_QWORD *)(v6 + 8) = v7;
    v8 = v13;
    v13 = 0;
    *(_QWORD *)(v6 + 16) = v8;
    v9 = v14;
    v14 = 0;
    *(_QWORD *)(v6 + 24) = v9;
    v10 = v15;
    v15 = 0;
    *(_QWORD *)(v6 + 32) = v10;
    *(_DWORD *)(v6 + 56) = v18;
  }
  v11 = v6;
  sub_2353900(a1, (unsigned __int64 *)&v11);
  if ( v11 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
  sub_233F7F0((__int64)&v13);
  if ( v12 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
}
