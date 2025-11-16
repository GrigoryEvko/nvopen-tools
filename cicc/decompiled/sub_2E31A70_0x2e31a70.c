// Function: sub_2E31A70
// Address: 0x2e31a70
//
__int64 __fastcall sub_2E31A70(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  unsigned int v4; // r8d

  v1 = *(_QWORD *)(a1 + 112);
  v2 = v1 + 8LL * *(unsigned int *)(a1 + 120);
  if ( v1 == v2 )
  {
    return 0;
  }
  else
  {
    do
    {
      v4 = *(unsigned __int8 *)(*(_QWORD *)v1 + 216LL);
      if ( (_BYTE)v4 )
        break;
      v1 += 8;
    }
    while ( v2 != v1 );
  }
  return v4;
}
