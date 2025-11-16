// Function: sub_EC73D0
// Address: 0xec73d0
//
__int64 __fastcall sub_EC73D0(__int64 a1, _DWORD *a2)
{
  _DWORD *v3; // rdi
  __int64 v4; // rdi
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+20h] [rbp-20h]
  char v8; // [rsp+21h] [rbp-1Fh]

  *a2 = 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    return 0;
  v3 = *(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8);
  if ( *v3 == 2 && sub_EC5140((__int64)v3) )
    return 0;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
    return sub_EC7220(a1, a2, "OS update");
  v4 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v6 = "invalid OS update specifier, comma expected";
  v7 = 3;
  return sub_ECE0E0(v4, &v6, 0, 0);
}
