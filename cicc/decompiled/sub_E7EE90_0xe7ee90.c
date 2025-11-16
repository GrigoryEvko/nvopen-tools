// Function: sub_E7EE90
// Address: 0xe7ee90
//
__int64 __fastcall sub_E7EE90(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v10; // [rsp+8h] [rbp-28h] BYREF
  __int64 v11; // [rsp+10h] [rbp-20h] BYREF
  __int64 v12[3]; // [rsp+18h] [rbp-18h] BYREF

  v4 = *a2;
  *a2 = 0;
  v10 = v4;
  v5 = *a3;
  *a3 = 0;
  v11 = v5;
  v6 = *a4;
  *a4 = 0;
  v12[0] = v6;
  v7 = sub_22077B0(6624);
  v8 = v7;
  if ( v7 )
    sub_E7DD10(v7, a1, &v10, &v11, v12);
  if ( v12[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v12[0] + 8LL))(v12[0]);
  if ( v11 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
  if ( v10 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
  return v8;
}
