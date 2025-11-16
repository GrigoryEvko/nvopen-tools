// Function: sub_70A250
// Address: 0x70a250
//
void __fastcall sub_70A250(int *a1, int a2)
{
  int v2; // r10d
  int v3; // edx
  int v4; // esi
  int v5; // eax
  int v6; // r9d
  int i; // r8d
  unsigned int v8; // eax
  unsigned int *j; // rdx
  unsigned int v10; // esi

  v2 = a2;
  if ( a2 > 31 )
  {
    v3 = a1[3];
    v4 = a1[2];
    v5 = v2;
    v6 = a1[1];
    for ( i = *a1; ; i = 0 )
    {
      if ( v3 )
        a1[4] = 1;
      v5 -= 32;
      v3 = v4;
      if ( v5 <= 31 )
        break;
      v4 = v6;
      v6 = i;
    }
    a1[3] = v4;
    v2 &= 0x1Fu;
    a1[2] = v6;
    a1[1] = i;
    *a1 = 0;
  }
  if ( v2 )
  {
    v8 = a1[3];
    if ( v8 << (32 - v2) )
      a1[4] = 1;
    for ( j = (unsigned int *)(a1 + 2); ; --j )
    {
      v10 = *j;
      j[1] = (*j << (32 - v2)) | (v8 >> v2);
      if ( a1 == (int *)j )
        break;
      v8 = v10;
    }
    *a1 = (unsigned int)*a1 >> v2;
  }
}
