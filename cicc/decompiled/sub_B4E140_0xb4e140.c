// Function: sub_B4E140
// Address: 0xb4e140
//
__int64 __fastcall sub_B4E140(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  __int64 v7; // rdx
  unsigned int v8; // r8d
  int v9; // r9d
  __int64 v11; // rcx
  _DWORD *v12; // rdi
  _DWORD *v13; // rax
  unsigned int v14; // r8d

  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  v9 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v9 - 17) > 1 || *(_QWORD *)(a2 + 8) != v7 )
    return v8;
  v11 = a4;
  v12 = &a3[v11];
  if ( a3 != &a3[v11] )
  {
    v13 = a3;
    while ( *v13 == -1 || 2 * *(_DWORD *)(v7 + 32) > *v13 )
    {
      if ( v12 == ++v13 )
        goto LABEL_9;
    }
    return 0;
  }
LABEL_9:
  v8 = 1;
  if ( (_BYTE)v9 != 18 )
    return v8;
  v8 = 0;
  if ( (unsigned int)(*a3 + 1) > 1 )
    return v8;
  v8 = 1;
  if ( a3 == v12 || v11 == 1 )
    return v8;
  LOBYTE(v14) = memcmp(a3 + 1, a3, v11 * 4 - 4) == 0;
  return v14;
}
