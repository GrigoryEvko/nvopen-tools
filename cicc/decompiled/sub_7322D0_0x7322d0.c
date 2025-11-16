// Function: sub_7322D0
// Address: 0x7322d0
//
__int64 __fastcall sub_7322D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD v7[10]; // [rsp-E8h] [rbp-E8h] BYREF
  unsigned int v8; // [rsp-98h] [rbp-98h]
  int v9; // [rsp-94h] [rbp-94h]

  if ( dword_4F077C4 != 2 )
    return 0;
  sub_76C7C0(v7, a2, &dword_4F077C4, a4, a5, a6);
  v7[2] = sub_72FB40;
  v7[8] = sub_728720;
  v9 = 1;
  sub_76D560(a1, v7);
  return v8;
}
