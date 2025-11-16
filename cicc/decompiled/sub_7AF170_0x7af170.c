// Function: sub_7AF170
// Address: 0x7af170
//
__int64 *__fastcall sub_7AF170(__int64 a1)
{
  __int64 *result; // rax

  result = *(__int64 **)(a1 + 16);
  if ( !result )
    goto LABEL_5;
  if ( unk_4F06498 > (unsigned __int64)result || unk_4F06490 <= (unsigned __int64)result )
  {
    result = sub_7AEFF0(*(_QWORD *)(a1 + 16));
LABEL_5:
    *(_BYTE *)(a1 + 48) |= 4u;
    *(_QWORD *)(a1 + 24) = result;
    return result;
  }
  *(_BYTE *)(a1 + 48) |= 4u;
  *(_QWORD *)(a1 + 24) = 0;
  return 0;
}
