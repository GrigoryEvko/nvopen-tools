// Function: sub_19CEB00
// Address: 0x19ceb00
//
__int64 __fastcall sub_19CEB00(__int64 a1, __int64 a2)
{
  int v2; // ecx
  __int64 result; // rax

  v2 = *(unsigned __int16 *)(a2 + 18) >> 1;
  result = (unsigned int)(1 << v2 >> 1);
  if ( !(1 << v2 >> 1) )
    return sub_15A9FE0(a1, **(_QWORD **)(a2 - 48));
  return result;
}
