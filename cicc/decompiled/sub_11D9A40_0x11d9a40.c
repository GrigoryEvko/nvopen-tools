// Function: sub_11D9A40
// Address: 0x11d9a40
//
__int64 __fastcall sub_11D9A40(__int64 a1, __int64 a2, char a3, __int64 *a4, __int64 a5, __int64 *a6)
{
  __int64 result; // rax
  bool v9; // r8
  __int64 v10; // [rsp+8h] [rbp-68h] BYREF
  __int64 v11; // [rsp+18h] [rbp-58h]
  char *v12; // [rsp+20h] [rbp-50h] BYREF
  char v13; // [rsp+40h] [rbp-30h]
  char v14; // [rsp+41h] [rbp-2Fh]

  v10 = a2;
  if ( a3 )
  {
    BYTE4(v11) = 0;
    v14 = 1;
    v12 = "sqrt";
    v13 = 3;
    return sub_B33BC0(a5, 0x14Fu, a1, v11, (__int64)&v12);
  }
  else
  {
    v9 = sub_11C9D70(a4, a6, *(_QWORD *)(a1 + 8), 0x1C0u, 0x1C1u, 0x1C2u);
    result = 0;
    if ( v9 )
      return sub_11CCA60(a1, a6, 0x1C0u, 0x1C1u, 0x1C2u, a5, &v10);
  }
  return result;
}
