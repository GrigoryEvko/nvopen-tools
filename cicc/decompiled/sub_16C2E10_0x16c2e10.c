// Function: sub_16C2E10
// Address: 0x16c2e10
//
__int64 __fastcall sub_16C2E10(__int64 a1)
{
  __int64 v1; // rax
  int v3; // eax
  const char *v4; // [rsp+0h] [rbp-50h] BYREF
  char v5; // [rsp+10h] [rbp-40h]
  char v6; // [rsp+11h] [rbp-3Fh]
  _QWORD v7[2]; // [rsp+20h] [rbp-30h] BYREF
  char v8; // [rsp+30h] [rbp-20h]

  sub_16C81E0();
  v6 = 1;
  v4 = "<stdin>";
  v5 = 3;
  sub_16C2770((__int64)v7, 0, (__int64)&v4);
  if ( (v8 & 1) != 0 )
  {
    v3 = v7[0];
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v3;
    *(_QWORD *)(a1 + 8) = v7[1];
  }
  else
  {
    v1 = v7[0];
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v1;
  }
  return a1;
}
