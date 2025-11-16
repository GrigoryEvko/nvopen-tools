// Function: sub_B8A550
// Address: 0xb8a550
//
__int64 __fastcall sub_B8A550(__int64 a1, unsigned __int64 a2)
{
  unsigned int v2; // r13d
  unsigned int v3; // r12d
  __int64 v5; // rdi
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned int v8; // r12d
  __int64 v9; // rdi

  v2 = 0;
  v3 = 0;
  sub_B80C30(a1 + 568);
  if ( *(_DWORD *)(a1 + 608) )
  {
    do
    {
      v5 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * v3);
      if ( v5 )
        v5 -= 176;
      ++v3;
      v2 |= sub_B89FF0(v5, a2);
      v6 = sub_B2BE50(a2);
      sub_B6EAA0(v6);
      v7 = *(_DWORD *)(a1 + 608);
    }
    while ( v7 > v3 );
    if ( v7 )
    {
      v8 = 0;
      do
      {
        v9 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * v8);
        if ( v9 )
          v9 -= 176;
        ++v8;
        sub_B82370(v9);
      }
      while ( v8 < *(_DWORD *)(a1 + 608) );
    }
  }
  *(_BYTE *)(a1 + 1288) = 1;
  return v2;
}
