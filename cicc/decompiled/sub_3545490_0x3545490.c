// Function: sub_3545490
// Address: 0x3545490
//
__int64 __fastcall sub_3545490(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r9
  __int64 v3; // r11
  __int64 v4; // r10
  __int64 v5; // r12
  unsigned int v6; // ebx
  __int64 v7; // rax

  v1 = *(int *)(a1 + 480);
  if ( (int)v1 <= 0 )
    return 0;
  v2 = 4 * v1;
  v3 = 0;
  v4 = 0;
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_DWORD *)(v5 + 48);
  do
  {
    if ( v6 > 1 )
    {
      v7 = 0;
      while ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 112) + v3) + v7 + 8) <= (unsigned __int64)*(unsigned int *)(*(_QWORD *)(v5 + 32) + 4 * v7 + 40) )
      {
        v7 += 8;
        if ( 8LL * (v6 - 2) + 8 == v7 )
          goto LABEL_9;
      }
      return 1;
    }
LABEL_9:
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 272) + v4) > *(_DWORD *)(a1 + 484) )
      return 1;
    v4 += 4;
    v3 += 144;
  }
  while ( v2 != v4 );
  return 0;
}
