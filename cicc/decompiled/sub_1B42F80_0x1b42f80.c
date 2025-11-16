// Function: sub_1B42F80
// Address: 0x1b42f80
//
__int64 __fastcall sub_1B42F80(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  if ( !*(_QWORD *)a1 )
    BUG();
  result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( !result )
    BUG();
  v2 = 0;
  if ( *(_BYTE *)(result - 8) == 77 )
    v2 = result - 24;
  *(_QWORD *)a1 = v2;
  return result;
}
