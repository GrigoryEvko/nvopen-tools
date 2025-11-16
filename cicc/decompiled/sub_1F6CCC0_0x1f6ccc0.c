// Function: sub_1F6CCC0
// Address: 0x1f6ccc0
//
__int64 __fastcall sub_1F6CCC0(__int64 a1)
{
  __int16 v1; // dx
  __int64 result; // rax

  v1 = *(_WORD *)(a1 + 24);
  result = a1;
  if ( (unsigned __int16)(v1 - 12) > 1u && (unsigned __int16)(v1 - 34) > 1u )
    return 0;
  return result;
}
