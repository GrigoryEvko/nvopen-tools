// Function: sub_1D47920
// Address: 0x1d47920
//
__int64 __fastcall sub_1D47920(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v4; // ecx
  unsigned __int64 v5; // rcx
  __int64 v6; // rdx
  int v7; // edx
  int v8; // ecx
  int v9; // r8d
  __int64 v10; // rdi
  unsigned int v11; // edx
  __int64 v12; // rsi
  __int64 v13; // rdx

  result = sub_15F3040(a1);
  if ( (_BYTE)result )
    return 0;
  v4 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned int)(v4 - 25) > 9 )
  {
    if ( (_BYTE)v4 == 78 )
    {
      v13 = *(_QWORD *)(a1 - 24);
      if ( !*(_BYTE *)(v13 + 16)
        && (*(_BYTE *)(v13 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v13 + 36) - 35) <= 3 )
      {
        return result;
      }
    }
    else
    {
      v5 = (unsigned int)(v4 - 34);
      if ( (unsigned int)v5 <= 0x36 )
      {
        v6 = 0x40018000000001LL;
        if ( _bittest64(&v6, v5) )
          return result;
      }
    }
    v7 = *(_DWORD *)(a2 + 232);
    if ( v7 )
    {
      v8 = v7 - 1;
      v9 = 1;
      v10 = *(_QWORD *)(a2 + 216);
      v11 = (v7 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v12 = *(_QWORD *)(v10 + 16LL * v11);
      if ( a1 == v12 )
        return result;
      while ( v12 != -8 )
      {
        v11 = v8 & (v9 + v11);
        v12 = *(_QWORD *)(v10 + 16LL * v11);
        if ( a1 == v12 )
          return result;
        ++v9;
      }
    }
    return 1;
  }
  return result;
}
