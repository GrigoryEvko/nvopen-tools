// Function: sub_1E15AF0
// Address: 0x1e15af0
//
__int64 __fastcall sub_1E15AF0(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 i; // r12
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 j; // rax

  v1 = 0;
  for ( i = *(_QWORD *)(a1 + 328); a1 + 320 != i; i = *(_QWORD *)(i + 8) )
  {
    v3 = *(_QWORD *)(i + 32);
    v4 = i + 24;
    if ( i + 24 != v3 )
    {
      for ( j = *(_QWORD *)(v3 + 8); v4 != j; v1 = 1 )
      {
        while ( (*(_BYTE *)(j + 46) & 4) == 0 )
        {
          j = *(_QWORD *)(j + 8);
          if ( v4 == j )
            goto LABEL_8;
        }
        j = sub_1E15AA0(i, *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL);
      }
    }
LABEL_8:
    ;
  }
  return v1;
}
