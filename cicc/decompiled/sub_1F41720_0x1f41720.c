// Function: sub_1F41720
// Address: 0x1f41720
//
__int64 __fastcall sub_1F41720(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // rbp
  char v6; // [rsp-28h] [rbp-28h] BYREF
  __int64 v7; // [rsp-8h] [rbp-8h]

  if ( *(_BYTE *)(a1 + 16) )
    return 0;
  v7 = v4;
  sub_1F40D10((__int64)&v6, a1, a2, a3, a4);
  return (v6 != 4) & (unsigned __int8)((v6 & 0xFB) != 2);
}
