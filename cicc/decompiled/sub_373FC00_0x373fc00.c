// Function: sub_373FC00
// Address: 0x373fc00
//
unsigned __int64 __fastcall sub_373FC00(__int64 *a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  unsigned __int8 *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rcx

  v5 = sub_37394D0(a1, a2, a4);
  v9 = (unsigned __int64)v5;
  if ( a3 )
  {
    v10 = sub_373DE40((__int64)a1, a3, (__int64)v5, v6, v7, v8);
    if ( v10 )
      sub_32494F0(a1, v9, 100, v10);
  }
  return v9;
}
