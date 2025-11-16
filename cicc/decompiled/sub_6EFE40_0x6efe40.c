// Function: sub_6EFE40
// Address: 0x6efe40
//
__int64 __fastcall sub_6EFE40(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 (__fastcall *v7)(__int64, _DWORD *); // [rsp-F8h] [rbp-F8h] BYREF
  unsigned int v8; // [rsp-A8h] [rbp-A8h]
  __int64 v9; // [rsp-98h] [rbp-98h]
  int v10; // [rsp-64h] [rbp-64h]

  *a2 = 0;
  if ( (*(_BYTE *)(a1 + 25) & 3) != 0 )
    return 0;
  if ( !(unsigned int)sub_8D3A70(*(_QWORD *)a1) )
    return 0;
  sub_76C7C0(&v7, a2, v3, v4, v5, v6);
  v7 = sub_6DEEF0;
  v9 = 0x100000001LL;
  sub_76CDC0(a1);
  result = v8;
  *a2 = v10;
  return result;
}
