// Function: sub_227B8F0
// Address: 0x227b8f0
//
__int64 __fastcall sub_227B8F0(__int64 a1)
{
  __int64 v1; // rcx
  __int64 result; // rax
  __int64 v3; // rdx

  v1 = *(_QWORD *)(a1 + 16);
  result = *(_QWORD *)(a1 + 8);
  while ( 1 )
  {
    v3 = result - 24;
    if ( !result )
      v3 = 0;
    if ( v1 != v3 + 48 )
      break;
    result = *(_QWORD *)(result + 8);
    *(_QWORD *)(a1 + 8) = result;
    if ( *(_QWORD *)a1 == result )
      break;
    if ( !result )
      BUG();
    v1 = *(_QWORD *)(result + 32);
    *(_WORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 16) = v1;
  }
  return result;
}
