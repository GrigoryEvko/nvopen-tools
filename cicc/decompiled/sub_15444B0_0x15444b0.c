// Function: sub_15444B0
// Address: 0x15444b0
//
__int64 __fastcall sub_15444B0(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 result; // rax
  int v5; // eax
  int v6; // edx
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // r12
  int v10; // edi

  v3 = a2;
  result = sub_1543FA0(a1, *(char **)a2);
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
  {
    v5 = *(_DWORD *)(a1 + 104);
    if ( v5 )
    {
      v6 = v5 - 1;
      v7 = *(_QWORD *)(a1 + 88);
      result = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v8 = *(_QWORD *)(v7 + 16 * result);
      if ( v3 == v8 )
        return result;
      v10 = 1;
      while ( v8 != -8 )
      {
        result = v6 & (unsigned int)(v10 + result);
        v8 = *(_QWORD *)(v7 + 16LL * (unsigned int)result);
        if ( v3 == v8 )
          return result;
        ++v10;
      }
    }
    result = 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF);
    if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
    {
      v9 = *(_QWORD *)(v3 - 8);
      v3 = v9 + result;
    }
    else
    {
      v9 = v3 - result;
    }
    for ( ; v3 != v9; v9 += 24 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v9 + 16LL) != 18 )
        result = sub_15444B0(a1);
    }
  }
  return result;
}
