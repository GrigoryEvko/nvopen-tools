// Function: sub_BA95A0
// Address: 0xba95a0
//
void __fastcall sub_BA95A0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // ebx

  v2 = *(_QWORD *)a1;
  if ( v2 )
  {
    v3 = *(_DWORD *)(a1 + 8);
    do
    {
      if ( (unsigned int)sub_B91A00(v2) <= v3 )
        break;
      if ( *(_DWORD *)(sub_BA9590(a1) + 32) )
        break;
      v2 = *(_QWORD *)a1;
      v3 = *(_DWORD *)(a1 + 8) + 1;
      *(_DWORD *)(a1 + 8) = v3;
    }
    while ( v2 );
  }
}
