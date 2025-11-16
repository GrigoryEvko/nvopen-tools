// Function: sub_82C230
// Address: 0x82c230
//
__int64 __fastcall sub_82C230(_BYTE *a1)
{
  __int64 result; // rax

  result = 0;
  if ( a1[16] )
    result = *(_QWORD *)(*(_QWORD *)a1 + 8LL);
  *(_QWORD *)a1 = result;
  return result;
}
