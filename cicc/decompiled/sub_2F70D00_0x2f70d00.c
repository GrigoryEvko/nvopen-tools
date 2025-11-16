// Function: sub_2F70D00
// Address: 0x2f70d00
//
void __fastcall sub_2F70D00(__int64 *a1, __int64 *a2, unsigned __int64 a3)
{
  __int64 *v3; // r13
  __int64 v5; // r15
  __int64 i; // r12
  unsigned int v7; // edx
  unsigned int v8; // eax
  __int64 v9; // rcx
  unsigned __int64 v10; // r8

  v3 = a2;
  if ( (char *)a2 - (char *)a1 > 16 )
  {
    v5 = ((char *)a2 - (char *)a1) >> 4;
    for ( i = (v5 - 2) / 2; ; --i )
    {
      sub_2F621E0((__int64)a1, i, v5, a1[2 * i], a1[2 * i + 1]);
      if ( !i )
        break;
    }
  }
  if ( a3 > (unsigned __int64)a2 )
  {
    do
    {
      v7 = *(_DWORD *)((*v3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v3 >> 1) & 3;
      v8 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a1 >> 1) & 3;
      if ( v7 < v8 || v7 <= v8 && v3[1] < (unsigned __int64)a1[1] )
      {
        v9 = *v3;
        v10 = v3[1];
        *v3 = *a1;
        v3[1] = a1[1];
        sub_2F621E0((__int64)a1, 0, ((char *)a2 - (char *)a1) >> 4, v9, v10);
      }
      v3 += 2;
    }
    while ( a3 > (unsigned __int64)v3 );
  }
}
