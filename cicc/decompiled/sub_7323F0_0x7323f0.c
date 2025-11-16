// Function: sub_7323F0
// Address: 0x7323f0
//
__int64 __fastcall sub_7323F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD v7[10]; // [rsp-E8h] [rbp-E8h] BYREF
  unsigned int v8; // [rsp-98h] [rbp-98h]
  int v9; // [rsp-94h] [rbp-94h]
  __int64 v10; // [rsp-90h] [rbp-90h]

  if ( dword_4F077C4 != 2 )
    return 0;
  sub_76C7C0(v7, a2, &dword_4F077C4, a4, a5, a6);
  v7[0] = sub_728260;
  v7[2] = sub_728240;
  v7[8] = sub_728430;
  v10 = 0x100000001LL;
  v9 = 1;
  sub_76D560(a1, v7);
  return v8;
}
