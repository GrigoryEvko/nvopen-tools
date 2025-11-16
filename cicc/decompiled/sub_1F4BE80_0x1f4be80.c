// Function: sub_1F4BE80
// Address: 0x1f4be80
//
__int64 __fastcall sub_1F4BE80(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  result = sub_38D7300(
             a1,
             *(_QWORD *)(a1 + 176),
             *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL) + ((unsigned __int64)a2 << 6) + 6));
  if ( (int)result < 0 )
    return 1000;
  return result;
}
