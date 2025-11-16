// Function: sub_892350
// Address: 0x892350
//
__int64 __fastcall sub_892350(__int64 a1)
{
  _QWORD *v1; // rdx
  __int64 result; // rax

  v1 = *(_QWORD **)(a1 + 216);
  result = v1[1];
  if ( !result )
    return *v1;
  return result;
}
