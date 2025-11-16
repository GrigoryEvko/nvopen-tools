// Function: sub_C7DF90
// Address: 0xc7df90
//
__int64 __fastcall sub_C7DF90(__int64 a1)
{
  unsigned int v1; // eax
  __int64 v2; // rax
  int v4; // eax
  _QWORD v5[2]; // [rsp+0h] [rbp-60h] BYREF
  char v6; // [rsp+10h] [rbp-50h]
  const char *v7; // [rsp+20h] [rbp-40h] BYREF
  char v8; // [rsp+40h] [rbp-20h]
  char v9; // [rsp+41h] [rbp-1Fh]

  sub_C87920(1);
  v9 = 1;
  v7 = "<stdin>";
  v8 = 3;
  v1 = sub_C83590();
  sub_C7DE70((__int64)v5, (__int64 *)v1, (__int64)&v7);
  if ( (v6 & 1) != 0 )
  {
    v4 = v5[0];
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v4;
    *(_QWORD *)(a1 + 8) = v5[1];
  }
  else
  {
    v2 = v5[0];
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v2;
  }
  return a1;
}
