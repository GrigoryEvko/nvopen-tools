// Function: sub_C1F520
// Address: 0xc1f520
//
__int64 __fastcall sub_C1F520(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  int v6; // eax
  __int64 v7; // rdx
  _QWORD v8[2]; // [rsp+0h] [rbp-60h] BYREF
  char v9; // [rsp+10h] [rbp-50h]
  _QWORD v10[2]; // [rsp+20h] [rbp-40h] BYREF
  __int64 v11; // [rsp+30h] [rbp-30h] BYREF

  sub_CA0F50(v10, a2);
  if ( (unsigned int)sub_2241AC0(v10, "-") )
    sub_CA4130((unsigned int)v8, a3, a2, -1, 1, 0, 1);
  else
    sub_C7DF90(v8);
  if ( (__int64 *)v10[0] != &v11 )
    j_j___libc_free_0(v10[0], v11 + 1);
  if ( (v9 & 1) != 0 && (v6 = v8[0], v7 = v8[1], LODWORD(v8[0])) )
  {
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v6;
    *(_QWORD *)(a1 + 8) = v7;
  }
  else
  {
    v4 = v8[0];
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v4;
  }
  return a1;
}
