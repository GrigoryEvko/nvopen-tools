// Function: sub_1BBA1B0
// Address: 0x1bba1b0
//
__int64 __fastcall sub_1BBA1B0(_QWORD *a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rcx
  int v4; // eax
  int v5; // edx
  __int64 v6; // r8
  __int64 v7; // rcx
  int v8; // edx

  v2 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)v2 <= 0x17u )
    return *a1;
  v3 = a1[1];
  v4 = v2 - 24;
  v5 = 0;
  if ( v3 )
    v5 = *(unsigned __int8 *)(v3 + 16) - 24;
  v6 = a2;
  if ( v4 != v5 )
  {
    v7 = a1[2];
    v8 = 0;
    if ( v7 )
      v8 = *(unsigned __int8 *)(v7 + 16) - 24;
    v6 = a2;
    if ( v4 != v8 )
      return *a1;
  }
  return v6;
}
