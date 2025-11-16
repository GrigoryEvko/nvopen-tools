// Function: sub_EA1E70
// Address: 0xea1e70
//
__int64 __fastcall sub_EA1E70(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v10; // [rsp+8h] [rbp-48h] BYREF
  __int64 v11; // [rsp+10h] [rbp-40h] BYREF
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *a2;
  *a2 = 0;
  v5 = *a3;
  *a3 = 0;
  v6 = *a4;
  *a4 = 0;
  v7 = sub_22077B0(448);
  v8 = v7;
  if ( v7 )
  {
    v12[0] = v6;
    v11 = v5;
    v10 = v4;
    sub_E8A5F0(v7, a1, &v10, &v11, v12);
    if ( v10 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
    if ( v11 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
    if ( v12[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v12[0] + 8LL))(v12[0]);
    *(_BYTE *)(v8 + 440) = 0;
    *(_QWORD *)v8 = &unk_49E4258;
  }
  else
  {
    if ( v6 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
    if ( v5 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
    if ( v4 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  }
  return v8;
}
