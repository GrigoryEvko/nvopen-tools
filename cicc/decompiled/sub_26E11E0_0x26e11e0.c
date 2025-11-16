// Function: sub_26E11E0
// Address: 0x26e11e0
//
unsigned __int64 __fastcall sub_26E11E0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  int *v3; // r14
  size_t v4; // r12
  size_t v6[2]; // [rsp+0h] [rbp-150h] BYREF
  __int64 v7; // [rsp+10h] [rbp-140h] BYREF
  __int128 v8; // [rsp+18h] [rbp-138h]
  __int128 v9; // [rsp+28h] [rbp-128h]
  __int128 v10; // [rsp+38h] [rbp-118h]
  __int64 v11; // [rsp+48h] [rbp-108h]
  __int64 v12; // [rsp+50h] [rbp-100h]
  __int64 v13; // [rsp+58h] [rbp-F8h]
  __int64 v14; // [rsp+60h] [rbp-F0h]
  __int64 v15; // [rsp+68h] [rbp-E8h]
  __int64 v16; // [rsp+70h] [rbp-E0h]
  __int64 v17; // [rsp+78h] [rbp-D8h]
  __int64 v18; // [rsp+80h] [rbp-D0h]
  void (__fastcall *v19)(__int64, __int64); // [rsp+88h] [rbp-C8h]
  int v20[48]; // [rsp+90h] [rbp-C0h] BYREF

  v2 = *a1;
  v11 = 0;
  v3 = *(int **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = sub_C64CA0;
  v7 = v2;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  if ( v3 )
  {
    sub_C7D030(v20);
    sub_C7D280(v20, v3, v4);
    sub_C7D290(v20, v6);
    v4 = v6[0];
  }
  *(_QWORD *)&v8 = v4;
  return sub_AC25F0(&v7, 0x10u, (__int64)sub_C64CA0);
}
