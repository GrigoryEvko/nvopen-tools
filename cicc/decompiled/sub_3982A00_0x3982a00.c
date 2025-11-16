// Function: sub_3982A00
// Address: 0x3982a00
//
__int64 __fastcall sub_3982A00(__int64 **a1, __int64 a2)
{
  unsigned int v2; // r8d
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  __int64 *v5; // rbx
  unsigned int v6; // eax

  v2 = *((_DWORD *)a1 + 2);
  if ( v2 )
    return v2;
  if ( *a1 )
  {
    v3 = **a1;
    do
    {
      v4 = v3 & 0xFFFFFFFFFFFFFFF8LL;
      v5 = (__int64 *)v4;
      if ( !v4 )
        break;
      v6 = *((_DWORD *)a1 + 2) + sub_3982940(v4 + 8, a2);
      *((_DWORD *)a1 + 2) = v6;
      v2 = v6;
      v3 = *v5;
    }
    while ( (*v5 & 4) == 0 );
  }
  return v2;
}
