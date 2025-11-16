// Function: sub_DF99B0
// Address: 0xdf99b0
//
__int64 __fastcall sub_DF99B0(__int64 a1, __int64 a2)
{
  __int64 (*v2)(void); // rax
  __int64 result; // rax

  v2 = *(__int64 (**)(void))(**(_QWORD **)a1 + 248LL);
  if ( (char *)v2 != (char *)sub_DF67D0 )
    return v2();
  result = sub_A73ED0((_QWORD *)(a2 + 72), 6);
  if ( !(_BYTE)result )
    return sub_B49560(a2, 6);
  return result;
}
