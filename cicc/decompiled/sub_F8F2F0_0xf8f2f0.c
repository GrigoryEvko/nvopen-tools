// Function: sub_F8F2F0
// Address: 0xf8f2f0
//
__int64 __fastcall sub_F8F2F0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  if ( !*(_QWORD *)a1 )
    BUG();
  result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( !result )
    BUG();
  v2 = 0;
  if ( *(_BYTE *)(result - 24) == 84 )
    v2 = result - 24;
  *(_QWORD *)a1 = v2;
  return result;
}
