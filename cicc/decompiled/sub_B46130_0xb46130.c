// Function: sub_B46130
// Address: 0xb46130
//
__int64 __fastcall sub_B46130(__int64 a1, __int64 a2, char a3)
{
  _QWORD *v3; // r8
  unsigned int v4; // eax
  __int64 v5; // rcx
  __int64 v6; // r11
  _QWORD *v7; // rax
  _QWORD *v8; // r10
  unsigned int v10; // r8d

  if ( *(_BYTE *)a1 != *(_BYTE *)a2 )
    return 0;
  LODWORD(v3) = 0;
  v4 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( v4 != (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) || *(_QWORD *)(a1 + 8) != *(_QWORD *)(a2 + 8) )
    return 0;
  if ( v4 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v3 = *(_QWORD **)(a2 - 8);
      v5 = v4;
    }
    else
    {
      v5 = v4;
      v3 = (_QWORD *)(a2 - 32LL * v4);
    }
    v6 = 4 * v5;
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    {
      v7 = *(_QWORD **)(a1 - 8);
      v8 = &v7[v6];
    }
    else
    {
      v8 = (_QWORD *)a1;
      v7 = (_QWORD *)(a1 - v6 * 8);
    }
    while ( *v7 == *v3 )
    {
      v7 += 4;
      v3 += 4;
      if ( v7 == v8 )
      {
        if ( *(_BYTE *)a1 != 84 )
          return sub_B45D20(a1, a2, 0, a3, (unsigned int)v3);
        LOBYTE(v10) = memcmp(
                        (const void *)(*(_QWORD *)(a1 - 8) + 32LL * *(unsigned int *)(a1 + 72)),
                        (const void *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72)),
                        8 * v5) == 0;
        return v10;
      }
    }
    return 0;
  }
  return sub_B45D20(a1, a2, 0, a3, (unsigned int)v3);
}
