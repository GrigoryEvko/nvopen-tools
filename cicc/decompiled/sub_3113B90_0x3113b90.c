// Function: sub_3113B90
// Address: 0x3113b90
//
__int64 __fastcall sub_3113B90(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = 0;
  if ( *(_QWORD *)a2 && (result = 1, *(_BYTE *)(a1 + 8)) )
  {
    result = *(unsigned __int8 *)(*(_QWORD *)a2 + 12LL);
    **(_QWORD **)a1 += result;
  }
  else
  {
    **(_QWORD **)a1 += result;
  }
  return result;
}
