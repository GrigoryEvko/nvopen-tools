// Function: sub_28E8F30
// Address: 0x28e8f30
//
__int64 __fastcall sub_28E8F30(__int64 *a1, unsigned int a2, _BYTE *a3)
{
  unsigned int v4; // r12d
  __int64 v5; // r14
  int v6; // ebx
  int v7; // r15d
  _BYTE *v8; // rdi
  __int64 v9; // rax
  _BYTE *v10; // rdi
  __int64 v12; // rax

  v4 = a2 + 1;
  v5 = *a1;
  v6 = *((_DWORD *)a1 + 2);
  v7 = *(_DWORD *)(*a1 + 16LL * a2);
  if ( a2 + 1 != v6 )
  {
    do
    {
      v9 = v5 + 16LL * v4;
      if ( *(_DWORD *)v9 != v7 )
        break;
      v8 = *(_BYTE **)(v9 + 8);
      if ( v8 == a3 || *v8 > 0x1Cu && *a3 > 0x1Cu && sub_B46220((__int64)v8, (__int64)a3) )
        return v4;
      ++v4;
    }
    while ( v6 != v4 );
  }
  v4 = a2 - 1;
  if ( a2 )
  {
    do
    {
      v12 = v5 + 16LL * v4;
      if ( *(_DWORD *)v12 != v7 )
        break;
      v10 = *(_BYTE **)(v12 + 8);
      if ( v10 == a3 || *v10 > 0x1Cu && *a3 > 0x1Cu && sub_B46220((__int64)v10, (__int64)a3) )
        return v4;
    }
    while ( v4-- != 0 );
    return a2;
  }
  else
  {
    return 0;
  }
}
