// Function: sub_2561480
// Address: 0x2561480
//
__int64 __fastcall sub_2561480(__int64 a1, __int64 *a2)
{
  unsigned __int8 *v2; // r13
  unsigned __int8 v3; // cl
  unsigned __int64 v4; // rdx
  unsigned __int8 *v5; // r14
  unsigned __int8 v6; // cl
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi
  int v10; // edx
  int v11; // ecx
  unsigned int v12; // edx
  __int64 v13; // rsi
  int v14; // r8d
  unsigned __int8 *v15; // r14
  unsigned __int8 v16; // cl

  if ( (unsigned int)(*(_DWORD *)(a1 + 3552) - 2) > 1 )
  {
    v2 = sub_250CBE0(a2, (__int64)a2);
    v3 = sub_2509800(a2);
    if ( v3 <= 7u && ((1LL << v3) & 0xA8) != 0 )
    {
      v4 = *a2 & 0xFFFFFFFFFFFFFFFCLL;
      if ( (*a2 & 3) == 3 )
        v4 = *(_QWORD *)(v4 + 24);
      if ( **(_BYTE **)(v4 - 32) == 25 )
        return 0;
    }
    if ( (v3 & 0xFD) == 4 )
    {
      if ( (v2[32] & 0xFu) - 7 > 1 )
        return 0;
      v5 = sub_250CBE0(a2, (__int64)a2);
      v6 = sub_2509800(a2);
      if ( v6 <= 6u && ((1LL << v6) & 0x54) != 0 && !(unsigned __int8)sub_254F400(a1, v5) )
        return 0;
    }
    else
    {
      v15 = sub_250CBE0(a2, (__int64)a2);
      v16 = sub_2509800(a2);
      if ( v16 <= 6u && ((1LL << v16) & 0x54) != 0 && !(unsigned __int8)sub_254F400(a1, v15) )
        return 0;
      if ( !v2 )
        return 1;
    }
    if ( *(_BYTE *)(a1 + 4296) )
      return 1;
    if ( (unsigned __int8)sub_253A110(*(_QWORD *)(a1 + 200), (__int64)v2) )
      return 1;
    v7 = sub_25096F0(a2);
    v8 = *(_QWORD *)(a1 + 200);
    if ( !*(_DWORD *)(v8 + 40) )
      return 1;
    v9 = *(_QWORD *)(v8 + 8);
    v10 = *(_DWORD *)(v8 + 24);
    if ( v10 )
    {
      v11 = v10 - 1;
      v12 = (v10 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v13 = *(_QWORD *)(v9 + 8LL * v12);
      if ( v13 != v7 )
      {
        v14 = 1;
        while ( v13 != -4096 )
        {
          v12 = v11 & (v14 + v12);
          v13 = *(_QWORD *)(v9 + 8LL * v12);
          if ( v7 == v13 )
            return 1;
          ++v14;
        }
        return 0;
      }
      return 1;
    }
    return 0;
  }
  return 0;
}
