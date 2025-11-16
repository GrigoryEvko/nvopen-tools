// Function: sub_39F0A90
// Address: 0x39f0a90
//
__int64 __fastcall sub_39F0A90(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4, char a5)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v12; // [rsp+8h] [rbp-38h] BYREF
  __int64 v13; // [rsp+10h] [rbp-30h] BYREF
  __int64 v14[5]; // [rsp+18h] [rbp-28h] BYREF

  v6 = *a2;
  *a2 = 0;
  v12 = v6;
  v7 = *a3;
  *a3 = 0;
  v13 = v7;
  v8 = *a4;
  *a4 = 0;
  v14[0] = v8;
  v9 = sub_22077B0(0x178u);
  v10 = v9;
  if ( v9 )
    sub_39EF740(v9, a1, &v12, &v13, v14);
  if ( v14[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v14[0] + 8LL))(v14[0]);
  if ( v13 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
  if ( v12 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
  if ( a5 )
    *(_BYTE *)(*(_QWORD *)(v10 + 264) + 484LL) |= 1u;
  return v10;
}
