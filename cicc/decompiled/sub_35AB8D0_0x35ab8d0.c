// Function: sub_35AB8D0
// Address: 0x35ab8d0
//
__int64 __fastcall sub_35AB8D0(__int64 a1)
{
  __int16 *v1; // rax
  __int16 *v2; // rdx
  int v3; // eax
  __int16 v4; // cx
  int v5; // eax

  v1 = *(__int16 **)(a1 + 32);
  if ( *(__int16 **)(a1 + 80) == v1 )
    return 0;
  v2 = v1 + 1;
  *(_QWORD *)(a1 + 32) = v1 + 1;
  v3 = *v1;
  v4 = v3;
  v5 = *(_DWORD *)(a1 + 24) + v3;
  *(_DWORD *)(a1 + 24) = v5;
  if ( !v4 )
    v2 = 0;
  *(_WORD *)(a1 + 40) = v5;
  *(_QWORD *)(a1 + 32) = v2;
  return 1;
}
