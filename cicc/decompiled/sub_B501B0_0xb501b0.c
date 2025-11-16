// Function: sub_B501B0
// Address: 0xb501b0
//
__int64 __fastcall sub_B501B0(__int64 a1, unsigned int *a2, __int64 a3)
{
  unsigned int *v3; // rcx
  char v4; // dl
  unsigned __int64 v5; // rax

  v3 = &a2[a3];
  if ( a2 == v3 )
    return a1;
  while ( 1 )
  {
    v4 = *(_BYTE *)(a1 + 8);
    v5 = *a2;
    if ( v4 != 16 )
      break;
    if ( v5 >= *(_QWORD *)(a1 + 32) )
      return 0;
    a1 = *(_QWORD *)(a1 + 24);
LABEL_5:
    if ( v3 == ++a2 )
      return a1;
  }
  if ( v4 == 15 && (unsigned int)v5 < *(_DWORD *)(a1 + 12) )
  {
    a1 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v5);
    goto LABEL_5;
  }
  return 0;
}
