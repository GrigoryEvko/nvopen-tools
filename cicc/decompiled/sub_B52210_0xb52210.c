// Function: sub_B52210
// Address: 0xb52210
//
__int64 __fastcall sub_B52210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int16 a5)
{
  int v6; // r8d

  v6 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v6 - 17) <= 1 )
    LOBYTE(v6) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
  if ( (_BYTE)v6 == 12 )
    return sub_B51D30(47, a1, a2, a3, a4, a5);
  else
    return sub_B52190(a1, a2, a3, a4, a5);
}
