// Function: sub_2C14210
// Address: 0x2c14210
//
__int64 __fastcall sub_2C14210(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  signed __int64 v8; // rax
  __int64 v9; // r8

  if ( (unsigned __int8)sub_2C46C30(a1 + 96) )
    return sub_DFD270(*(_QWORD *)a3, 55, *(_DWORD *)(a3 + 176));
  v5 = sub_2BFD6A0(a3 + 16, a1 + 96);
  v6 = sub_2AAEDF0(v5, a2);
  v7 = sub_BCB2A0(*(_QWORD **)(a3 + 56));
  sub_2AAEDF0(v7, a2);
  v8 = sub_DFD2D0(*(__int64 **)a3, 57, v6);
  v9 = v8 * (((unsigned int)(*(_DWORD *)(a1 + 56) + 1) >> 1) - 1);
  if ( !is_mul_ok(v8, ((unsigned int)(*(_DWORD *)(a1 + 56) + 1) >> 1) - 1) )
  {
    if ( (unsigned int)(*(_DWORD *)(a1 + 56) + 1) >> 1 == 1 )
      return 0x8000000000000000LL;
    v9 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v8 <= 0 )
      return 0x8000000000000000LL;
  }
  return v9;
}
