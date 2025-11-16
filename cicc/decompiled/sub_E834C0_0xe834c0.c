// Function: sub_E834C0
// Address: 0xe834c0
//
__int64 __fastcall sub_E834C0(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 a5, char a6)
{
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v13; // [rsp+18h] [rbp-48h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h] BYREF
  __int64 v15[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = *a2;
  *a2 = 0;
  v7 = *a3;
  *a3 = 0;
  v8 = *a4;
  *a4 = 0;
  v9 = sub_22077B0(480);
  v10 = v9;
  if ( v9 )
  {
    v15[0] = v8;
    v14 = v7;
    v13 = v6;
    sub_E8A5F0(v9, a1, &v13, &v14, v15);
    if ( v13 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
    if ( v14 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
    if ( v15[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v15[0] + 8LL))(v15[0]);
    *(_QWORD *)(v10 + 448) = 0;
    *(_QWORD *)v10 = off_49E2A50;
    *(_QWORD *)(v10 + 456) = 0;
    *(_BYTE *)(v10 + 440) = a6;
    *(_QWORD *)(v10 + 464) = 0;
    *(_DWORD *)(v10 + 472) = 0;
  }
  else
  {
    if ( v8 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8);
    if ( v7 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
    if ( v6 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
  }
  return v10;
}
