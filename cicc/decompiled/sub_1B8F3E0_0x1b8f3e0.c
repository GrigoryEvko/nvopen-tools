// Function: sub_1B8F3E0
// Address: 0x1b8f3e0
//
__int64 __fastcall sub_1B8F3E0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ecx
  __int64 result; // rax
  unsigned int v4; // r9d
  __int64 v5; // rdi
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r8
  unsigned int v9; // eax
  int v10; // r11d
  __int64 v11; // r10
  int v12; // eax
  __int64 *v13; // rax
  int v14; // esi

  v2 = *(_DWORD *)(a1 + 72);
  result = 0;
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = *(_QWORD *)(a1 + 56);
    v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
      return v7[1];
    }
    else
    {
      v9 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = 1;
      if ( v8 == -8 )
      {
        return 0;
      }
      else
      {
        while ( 1 )
        {
          v9 = v4 & (v10 + v9);
          v11 = *(_QWORD *)(v5 + 16LL * v9);
          if ( a2 == v11 )
            break;
          ++v10;
          if ( v11 == -8 )
            return 0;
        }
        v12 = 1;
        while ( v8 != -8 )
        {
          v14 = v12 + 1;
          v6 = v4 & (v12 + v6);
          v13 = (__int64 *)(v5 + 16LL * v6);
          v8 = *v13;
          if ( v11 == *v13 )
            return v13[1];
          v12 = v14;
        }
        v13 = (__int64 *)(v5 + 16LL * v2);
        return v13[1];
      }
    }
  }
  return result;
}
