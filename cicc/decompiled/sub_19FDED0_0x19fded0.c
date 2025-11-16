// Function: sub_19FDED0
// Address: 0x19fded0
//
__int64 __fastcall sub_19FDED0(__int64 *a1, unsigned int a2, __int64 a3)
{
  unsigned int v5; // r12d
  __int64 v6; // rdx
  int v7; // ebx
  int v8; // r14d
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v13; // rax

  v5 = a2 + 1;
  v6 = *a1;
  v7 = *((_DWORD *)a1 + 2);
  v8 = *(_DWORD *)(*a1 + 16LL * a2);
  if ( a2 + 1 != v7 )
  {
    while ( 1 )
    {
      v10 = v6 + 16LL * v5;
      if ( *(_DWORD *)v10 != v8 )
        break;
      v9 = *(_QWORD *)(v10 + 8);
      if ( v9 == a3 || *(_BYTE *)(v9 + 16) > 0x17u && *(_BYTE *)(a3 + 16) > 0x17u && sub_15F41F0(v9, a3) )
        return v5;
      if ( ++v5 == v7 )
        break;
      v6 = *a1;
    }
  }
  v5 = a2 - 1;
  if ( a2 )
  {
    do
    {
      v13 = *a1 + 16LL * v5;
      if ( *(_DWORD *)v13 != v8 )
        break;
      v11 = *(_QWORD *)(v13 + 8);
      if ( v11 == a3 || *(_BYTE *)(v11 + 16) > 0x17u && *(_BYTE *)(a3 + 16) > 0x17u && sub_15F41F0(v11, a3) )
        return v5;
    }
    while ( v5-- != 0 );
    return a2;
  }
  else
  {
    return 0;
  }
}
