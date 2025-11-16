// Function: sub_38ECD60
// Address: 0x38ecd60
//
__int64 __fastcall sub_38ECD60(__int64 a1, __int64 *a2, _QWORD *a3)
{
  unsigned int v4; // eax
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned int v7; // r12d
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  char v10; // [rsp+10h] [rbp-30h]
  char v11; // [rsp+11h] [rbp-2Fh]

  v9[0] = 0;
  LOBYTE(v4) = sub_38EB6A0(a1, a2, (__int64)v9);
  v7 = v4;
  if ( (_BYTE)v4 )
    return v7;
  if ( **(_DWORD **)(a1 + 152) != 18 )
  {
    v11 = 1;
    v9[0] = "expected ')' in parentheses expression";
    v10 = 3;
    return (unsigned int)sub_3909CF0(a1, v9, 0, 0, v5, v6);
  }
  *a3 = sub_39092B0();
  sub_38EB180(a1);
  return v7;
}
