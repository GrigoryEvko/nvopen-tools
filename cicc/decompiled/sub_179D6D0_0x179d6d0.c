// Function: sub_179D6D0
// Address: 0x179d6d0
//
bool __fastcall sub_179D6D0(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 *v6; // r8
  int v7; // edx
  unsigned int v8; // eax
  __int64 *v9; // rdi
  __int64 v10; // r9
  int v11; // edi
  int v12; // r10d

  result = 0;
  if ( *(_BYTE *)(a3 + 16) == 35 && *(_BYTE *)(a2 + 16) == 47 )
  {
    v4 = *(_QWORD *)(a1 + 32);
    v5 = *(unsigned int *)(a1 + 48);
    v6 = (__int64 *)(v4 + 8 * v5);
    result = 1;
    if ( (_DWORD)v5 )
    {
      v7 = v5 - 1;
      v8 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v4 + 8LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
      {
        return v6 == v9;
      }
      else
      {
        v11 = 1;
        while ( v10 != -8 )
        {
          v12 = v11 + 1;
          v8 = v7 & (v11 + v8);
          v9 = (__int64 *)(v4 + 8LL * v8);
          v10 = *v9;
          if ( a2 == *v9 )
            return v6 == v9;
          v11 = v12;
        }
        return 1;
      }
    }
  }
  return result;
}
