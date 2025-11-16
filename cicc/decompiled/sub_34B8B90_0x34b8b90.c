// Function: sub_34B8B90
// Address: 0x34b8b90
//
__int64 __fastcall sub_34B8B90(__int64 a1, _DWORD *a2, _DWORD *a3, unsigned int a4)
{
  _DWORD *v5; // r15
  char v8; // al
  __int64 *v9; // r14
  __int64 v10; // rcx
  __int64 *v11; // r12
  __int64 i; // rbx
  __int64 v13; // rdi
  __int64 v15; // r12
  int v16; // eax
  int v17; // eax
  __int64 v18; // [rsp+8h] [rbp-38h]

  v5 = a2;
  if ( !a2 )
    goto LABEL_3;
  while ( 1 )
  {
    while ( 1 )
    {
      if ( a3 == v5 )
      {
        LODWORD(v10) = a4;
        return (unsigned int)v10;
      }
LABEL_3:
      v8 = *(_BYTE *)(a1 + 8);
      if ( v8 != 15 )
        break;
      v9 = *(__int64 **)(a1 + 16);
      v10 = a4;
      v11 = &v9[*(unsigned int *)(a1 + 12)];
      if ( v11 == v9 )
        return (unsigned int)v10;
      for ( i = 0; ; ++i )
      {
        a1 = *v9;
        if ( v5 )
        {
          if ( *v5 == i )
            break;
        }
        v13 = *v9++;
        LODWORD(v10) = sub_34B8B90(v13, 0, 0, v10);
        if ( v11 == v9 )
          return (unsigned int)v10;
      }
      ++v5;
      a4 = v10;
    }
    LODWORD(v10) = a4 + 1;
    if ( v8 != 16 )
      return (unsigned int)v10;
    v15 = *(_QWORD *)(a1 + 24);
    v18 = a1;
    v16 = sub_34B8B90(v15, 0, 0, 0);
    if ( !v5 )
      break;
    v17 = *v5 * v16;
    a1 = v15;
    ++v5;
    a4 += v17;
  }
  LODWORD(v10) = *(_DWORD *)(v18 + 32) * v16 + a4;
  return (unsigned int)v10;
}
