// Function: sub_1632FD0
// Address: 0x1632fd0
//
void __fastcall sub_1632FD0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // ebx

  v2 = *(_QWORD *)a1;
  if ( v2 )
  {
    v3 = *(_DWORD *)(a1 + 8);
    do
    {
      if ( (unsigned int)sub_161F520(v2) <= v3 )
        break;
      if ( *(_DWORD *)(sub_1632FC0(a1) + 36) )
        break;
      v2 = *(_QWORD *)a1;
      v3 = *(_DWORD *)(a1 + 8) + 1;
      *(_DWORD *)(a1 + 8) = v3;
    }
    while ( v2 );
  }
}
