// Function: sub_C8CA60
// Address: 0xc8ca60
//
__int64 *__fastcall sub_C8CA60(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdi
  int v4; // edx
  unsigned int v5; // eax
  __int64 *v6; // r8
  __int64 v7; // rcx
  int v9; // r8d
  int v10; // r9d

  v2 = *(_DWORD *)(a1 + 16);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = v2 - 1;
  v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v3 + 8LL * v5);
  v7 = *v6;
  if ( a2 != *v6 )
  {
    v9 = 1;
    while ( v7 != -1 )
    {
      v10 = v9 + 1;
      v5 = v4 & (v9 + v5);
      v6 = (__int64 *)(v3 + 8LL * v5);
      v7 = *v6;
      if ( *v6 == a2 )
        return v6;
      v9 = v10;
    }
    return 0;
  }
  return v6;
}
