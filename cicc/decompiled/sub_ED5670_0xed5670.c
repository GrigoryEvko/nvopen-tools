// Function: sub_ED5670
// Address: 0xed5670
//
void __fastcall sub_ED5670(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int *v3; // r15
  unsigned int v4; // r14d
  unsigned int v7; // esi
  unsigned __int8 *v8; // rax
  int v9; // edx
  int v10; // ecx
  __int64 v11; // rdx

  if ( *(_DWORD *)(a1 + 4) )
  {
    v3 = (unsigned int *)(a1 + 8);
    v4 = 0;
    do
    {
      sub_ED54D0(v3, a2, a3);
      v7 = v3[1];
      if ( v7 )
      {
        v8 = (unsigned __int8 *)(v3 + 2);
        v9 = 0;
        do
        {
          v10 = *v8++;
          v9 += v10;
        }
        while ( v8 != (unsigned __int8 *)((char *)v3 + v7 + 8) );
        v11 = ((v7 + 15) & 0xFFFFFFF8) + 16 * v9;
      }
      else
      {
        v11 = 8;
      }
      v3 = (unsigned int *)((char *)v3 + v11);
      ++v4;
    }
    while ( *(_DWORD *)(a1 + 4) > v4 );
  }
}
