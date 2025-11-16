// Function: sub_7D3880
// Address: 0x7d3880
//
_BOOL8 __fastcall sub_7D3880(unsigned __int8 a1)
{
  __int64 v1; // rdx
  _BOOL8 result; // rax

  v1 = qword_4D04A60[a1];
  result = 0;
  if ( v1 )
  {
    result = 1;
    if ( !*(_QWORD *)(v1 + 32) )
      return *(_QWORD *)(v1 + 24) != 0;
  }
  return result;
}
