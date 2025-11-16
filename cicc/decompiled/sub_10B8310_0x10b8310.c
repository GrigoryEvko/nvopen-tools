// Function: sub_10B8310
// Address: 0x10b8310
//
__int64 __fastcall sub_10B8310(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax

  v2 = *(_QWORD *)(a2 - 64);
  if ( !v2 )
    return 0;
  **a1 = v2;
  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 )
    return 0;
  *a1[1] = v3;
  return 1;
}
