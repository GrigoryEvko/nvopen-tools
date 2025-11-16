// Function: sub_39822D0
// Address: 0x39822d0
//
__int64 __fastcall sub_39822D0(__int64 a1, __int64 a2, __int16 a3)
{
  if ( (a3 & 0xFFF7) == 6 || a3 == 23 )
    return 4;
  else
    return *(unsigned int *)(*(_QWORD *)(a2 + 240) + 8LL);
}
