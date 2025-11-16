// Function: sub_19EE910
// Address: 0x19ee910
//
__int64 __fastcall sub_19EE910(__int64 a1, __int64 a2, __int64 a3)
{
  int v6; // eax
  int v7; // edx
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 v10; // rcx
  int v11; // edi
  char v12; // al
  __int64 v13; // rsi
  unsigned int v14; // r8d
  int v15; // eax
  int v16; // eax
  __int64 v17; // rax
  int v18; // esi
  _QWORD v19[12]; // [rsp+10h] [rbp-60h] BYREF

  v6 = *(_DWORD *)(a1 + 1856);
  if ( v6 )
  {
    v7 = v6 - 1;
    v8 = *(_QWORD *)(a1 + 1840);
    result = (v6 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v10 = *(_QWORD *)(v8 + 8 * result);
    if ( a3 == v10 )
      return result;
    v11 = 1;
    while ( v10 != -8 )
    {
      result = v7 & (unsigned int)(v11 + result);
      v10 = *(_QWORD *)(v8 + 8LL * (unsigned int)result);
      if ( a3 == v10 )
        return result;
      ++v11;
    }
  }
  result = *(unsigned int *)(a2 + 24);
  if ( !(_DWORD)result )
  {
    v12 = sub_19EB120(a1 + 1896, (__int64 *)(a2 + 40), v19);
    v13 = v19[0];
    if ( v12 )
      return sub_165A590((__int64)v19, v13 + 8, a3);
    v14 = *(_DWORD *)(a1 + 1920);
    v15 = *(_DWORD *)(a1 + 1912);
    ++*(_QWORD *)(a1 + 1896);
    v16 = v15 + 1;
    if ( 4 * v16 >= 3 * v14 )
    {
      v18 = 2 * v14;
    }
    else
    {
      if ( v14 - *(_DWORD *)(a1 + 1916) - v16 > v14 >> 3 )
      {
LABEL_10:
        *(_DWORD *)(a1 + 1912) = v16;
        if ( *(_QWORD *)v13 != -8 )
          --*(_DWORD *)(a1 + 1916);
        v17 = *(_QWORD *)(a2 + 40);
        *(_QWORD *)(v13 + 8) = 0;
        *(_QWORD *)(v13 + 32) = 2;
        *(_QWORD *)v13 = v17;
        *(_QWORD *)(v13 + 16) = v13 + 48;
        *(_QWORD *)(v13 + 24) = v13 + 48;
        *(_DWORD *)(v13 + 40) = 0;
        return sub_165A590((__int64)v19, v13 + 8, a3);
      }
      v18 = v14;
    }
    sub_19EE760(a1 + 1896, v18);
    sub_19EB120(a1 + 1896, (__int64 *)(a2 + 40), v19);
    v13 = v19[0];
    v16 = *(_DWORD *)(a1 + 1912) + 1;
    goto LABEL_10;
  }
  return result;
}
