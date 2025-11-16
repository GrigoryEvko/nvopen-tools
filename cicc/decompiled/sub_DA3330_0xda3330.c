// Function: sub_DA3330
// Address: 0xda3330
//
__int64 __fastcall sub_DA3330(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  _QWORD *v9; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  _QWORD *v16; // rax
  bool v17; // zf
  __int64 v18; // [rsp+0h] [rbp-70h]
  int v20; // [rsp+18h] [rbp-58h] BYREF
  int v21; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 v22; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v23; // [rsp+28h] [rbp-48h] BYREF
  __int64 v24; // [rsp+30h] [rbp-40h] BYREF
  _QWORD *v25; // [rsp+38h] [rbp-38h] BYREF

  if ( (unsigned __int8)sub_D98220(*a1, a2, &v23, &v22, &v20) )
  {
    v9 = v23;
  }
  else
  {
    v18 = *a1;
    v13 = sub_D95540(a2);
    v9 = sub_DA2C50(v18, v13, 0, 0);
    v22 = a2;
    v23 = v9;
    v20 = a6;
  }
  if ( *((_WORD *)v9 + 12) || a6 != (a6 & v20) )
    return 0;
  if ( (unsigned __int8)sub_D98220(*a1, a3, &v25, &v24, &v21) )
  {
    if ( *((_WORD *)v25 + 12) || a6 != (a6 & v21) )
      return 0;
    a3 = v24;
  }
  else
  {
    v14 = *a1;
    v15 = sub_D95540(a3);
    v16 = sub_DA2C50(v14, v15, 0, 0);
    v24 = a3;
    v17 = *((_WORD *)v16 + 12) == 0;
    v25 = v16;
    v21 = a6;
    if ( !v17 )
      return 0;
  }
  if ( v22 != a3 )
    return 0;
  v11 = v23[4];
  if ( *(_DWORD *)(a4 + 8) <= 0x40u && *(_DWORD *)(v11 + 32) <= 0x40u )
  {
    *(_QWORD *)a4 = *(_QWORD *)(v11 + 24);
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(v11 + 32);
  }
  else
  {
    sub_C43990(a4, v11 + 24);
  }
  v12 = v25[4];
  if ( *(_DWORD *)(a5 + 8) <= 0x40u && *(_DWORD *)(v12 + 32) <= 0x40u )
  {
    *(_QWORD *)a5 = *(_QWORD *)(v12 + 24);
    *(_DWORD *)(a5 + 8) = *(_DWORD *)(v12 + 32);
    return 1;
  }
  else
  {
    sub_C43990(a5, v12 + 24);
    return 1;
  }
}
