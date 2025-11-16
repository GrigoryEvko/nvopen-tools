// Function: sub_20C7BE0
// Address: 0x20c7be0
//
__int64 __fastcall sub_20C7BE0(__int64 a1, _DWORD *a2, _DWORD *a3, unsigned int a4)
{
  _DWORD *v6; // r15
  char v7; // al
  __int64 *v8; // r13
  __int64 *v9; // r14
  __int64 *v10; // rbx
  __int64 v11; // rdi
  __int64 v13; // r13
  int v14; // eax
  _DWORD *v15; // r8

  if ( a2 == a3 && a2 )
    return a4;
  v6 = a2;
  while ( 1 )
  {
    v7 = *(_BYTE *)(a1 + 8);
    if ( v7 == 13 )
    {
      v8 = *(__int64 **)(a1 + 16);
      v9 = &v8[*(unsigned int *)(a1 + 12)];
      if ( v9 == v8 )
        return a4;
      v10 = *(__int64 **)(a1 + 16);
      while ( !v6 || *v6 != (unsigned int)(v10 - v8) )
      {
        v11 = *v10++;
        a4 = sub_20C7BE0(v11, 0, 0, a4);
        if ( v10 == v9 )
          return a4;
      }
      a1 = *v10;
      v15 = v6 + 1;
      goto LABEL_14;
    }
    if ( v7 != 14 )
      return a4 + 1;
    v13 = *(_QWORD *)(a1 + 24);
    v14 = sub_20C7BE0(v13, 0, 0, 0);
    if ( !v6 )
      return a4 + *(_DWORD *)(a1 + 32) * v14;
    v15 = v6 + 1;
    a1 = v13;
    a4 += *v6 * v14;
LABEL_14:
    v6 = v15;
    if ( a3 == v15 )
      return a4;
  }
}
