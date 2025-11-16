// Function: sub_C49940
// Address: 0xc49940
//
__int64 __fastcall sub_C49940(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // rax

  while ( a3 )
  {
    v3 = (unsigned int)(a3 - 1);
    v4 = *(_QWORD *)(a1 + 8 * v3);
    --a3;
    v5 = *(_QWORD *)(a2 + 8 * v3);
    if ( v4 != v5 )
      return v5 < v4 ? 1 : -1;
  }
  return 0;
}
