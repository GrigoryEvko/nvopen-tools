// Function: sub_390A1E0
// Address: 0x390a1e0
//
__int64 __fastcall sub_390A1E0(__int64 a1, _DWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _BOOL8 v7; // r8
  __int64 v9; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v10[5]; // [rsp+8h] [rbp-28h] BYREF

  (*(void (__fastcall **)(__int64 *))(*(_QWORD *)a2 + 24LL))(&v9);
  if ( (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v9 + 16LL))(v9) != 2 )
    sub_16BD130("dwo only supported with ELF", 1u);
  v6 = v9;
  v7 = a2[4] == 1;
  v9 = 0;
  v10[0] = v6;
  sub_392ECA0(a1, v10, a3, a4, v7);
  if ( v10[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v10[0] + 8LL))(v10[0]);
  if ( v9 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
  return a1;
}
