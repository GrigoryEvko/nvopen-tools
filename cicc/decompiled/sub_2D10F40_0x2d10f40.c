// Function: sub_2D10F40
// Address: 0x2d10f40
//
__int64 __fastcall sub_2D10F40(__int64 a1, unsigned int *a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // esi
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx

  v2 = *(_QWORD *)(a1 + 16);
  v3 = a1 + 8;
  if ( v2 )
  {
    v4 = *a2;
    v5 = a1 + 8;
    do
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(v2 + 16);
        v7 = *(_QWORD *)(v2 + 24);
        if ( *(_DWORD *)(v2 + 32) >= v4 )
          break;
        v2 = *(_QWORD *)(v2 + 24);
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v2;
      v2 = *(_QWORD *)(v2 + 16);
    }
    while ( v6 );
LABEL_6:
    if ( v5 != v3 && v4 >= *(_DWORD *)(v5 + 32) )
      return v5;
  }
  return v3;
}
