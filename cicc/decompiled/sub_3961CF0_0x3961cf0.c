// Function: sub_3961CF0
// Address: 0x3961cf0
//
__int64 __fastcall sub_3961CF0(__int64 a1, __int64 a2, char a3)
{
  _QWORD *v4; // rax
  __int64 v5; // r8
  _QWORD *v6; // rdx
  unsigned __int8 v7; // al
  _QWORD *v8; // rcx
  __int64 v10; // rcx
  int v11; // eax

  v4 = sub_1648700(a2);
  v5 = 0;
  v6 = v4;
  v7 = *((_BYTE *)v4 + 16);
  if ( v7 <= 0x17u )
    return v5;
  if ( a3 && v7 == 78 )
  {
    v10 = *(v6 - 3);
    if ( *(_BYTE *)(v10 + 16) )
      return v6[5];
    if ( (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
      return v6[5];
    if ( (unsigned int)(*(_DWORD *)(v10 + 36) - 35) > 3 )
    {
      if ( (*(_BYTE *)(v10 + 33) & 0x20) == 0 )
        return v6[5];
      v11 = *(_DWORD *)(v10 + 36);
      if ( v11 != 4 && (unsigned int)(v11 - 116) > 1 )
        return v6[5];
    }
    return v5;
  }
  if ( v7 == 77 )
  {
    if ( (*((_BYTE *)v6 + 23) & 0x40) != 0 )
      v8 = (_QWORD *)*(v6 - 1);
    else
      v8 = &v6[-3 * (*((_DWORD *)v6 + 5) & 0xFFFFFFF)];
    return v8[3 * *((unsigned int *)v6 + 14) + 1 + -1431655765 * (unsigned int)((a2 - (__int64)v8) >> 3)];
  }
  return v6[5];
}
