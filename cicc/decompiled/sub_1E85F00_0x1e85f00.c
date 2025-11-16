// Function: sub_1E85F00
// Address: 0x1e85f00
//
__int64 __fastcall sub_1E85F00(__int64 a1, int a2)
{
  __int64 v3; // rsi

  if ( a2 >= 0 )
    return 0;
  v3 = a2 & 0x7FFFFFFF;
  if ( (unsigned int)v3 >= *(_DWORD *)(a1 + 336) )
    return 0;
  else
    return *(_QWORD *)(*(_QWORD *)(a1 + 328) + 8 * v3);
}
