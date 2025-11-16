// Function: sub_892330
// Address: 0x892330
//
__int64 __fastcall sub_892330(__int64 a1)
{
  __int64 v1; // rdx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 168);
  result = *(_QWORD *)(v1 + 176);
  if ( !result )
    return *(_QWORD *)(v1 + 168);
  return result;
}
