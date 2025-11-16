// Function: sub_2E943E0
// Address: 0x2e943e0
//
__int64 __fastcall sub_2E943E0(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 i; // r12
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 j; // rax

  v1 = 0;
  for ( i = *(_QWORD *)(a1 + 328); a1 + 320 != i; i = *(_QWORD *)(i + 8) )
  {
    v3 = *(_QWORD *)(i + 56);
    v4 = i + 48;
    if ( i + 48 != v3 )
    {
      for ( j = *(_QWORD *)(v3 + 8); v4 != j; v1 = 1 )
      {
        while ( (*(_BYTE *)(j + 44) & 4) == 0 )
        {
          j = *(_QWORD *)(j + 8);
          if ( v4 == j )
            goto LABEL_8;
        }
        j = sub_2E94390(i, *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL);
      }
    }
LABEL_8:
    ;
  }
  return v1;
}
