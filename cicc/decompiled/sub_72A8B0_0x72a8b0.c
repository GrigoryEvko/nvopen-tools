// Function: sub_72A8B0
// Address: 0x72a8b0
//
__int64 __fastcall sub_72A8B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ebx
  __int64 result; // rax
  _QWORD v8[10]; // [rsp+0h] [rbp-F0h] BYREF
  int v9; // [rsp+54h] [rbp-9Ch]

  v6 = dword_4F07AD0;
  dword_4F07AD0 = 0;
  sub_76C7C0(v8, a2, a3, a4, a5, a6);
  v8[0] = sub_72E6E0;
  v8[8] = sub_72E6C0;
  v8[2] = sub_72E970;
  v9 = 1;
  sub_76CDC0(a1);
  result = (unsigned int)dword_4F07AD0;
  dword_4F07AD0 = v6;
  return result;
}
