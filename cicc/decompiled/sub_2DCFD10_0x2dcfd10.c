// Function: sub_2DCFD10
// Address: 0x2dcfd10
//
__int64 __fastcall sub_2DCFD10(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9

  v3 = *a2;
  if ( (unsigned __int8)sub_BB98D0(a1, v3) || (*(_BYTE *)(*a2 + 3) & 0x40) == 0 )
    return 0;
  else
    return sub_2DCCA90(a2, v3, v4, v5, v6, v7);
}
