// Function: sub_2FDED80
// Address: 0x2fded80
//
bool __fastcall sub_2FDED80(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  int v3; // eax
  unsigned __int64 v4; // r12
  int v5; // eax
  __int64 v6; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rdx

  v2 = sub_2E319B0(a2, 1);
  if ( v2 == a2 + 48 )
    return 1;
  v3 = *(unsigned __int16 *)(v2 + 68);
  if ( v3 == 27 )
    return 0;
  if ( v3 == 36 )
    return 0;
  v4 = sub_2E31A10(a2, 1);
  if ( (*(_WORD *)(v4 + 68) & 0xFFFD) == 0x25 )
    return 0;
  if ( v2 == v4 )
    return 1;
  v5 = *(_DWORD *)(v4 + 44);
  if ( (v5 & 4) == 0 && (v5 & 8) != 0 )
    LOBYTE(v6) = sub_2E88A90(v4, 32, 1);
  else
    v6 = (*(_QWORD *)(*(_QWORD *)(v4 + 16) + 24LL) >> 5) & 1LL;
  if ( !(_BYTE)v6 )
    return 1;
  v8 = *(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v8 )
    BUG();
  v9 = *(_QWORD *)v8;
  if ( (*(_QWORD *)v8 & 4) == 0 && (*(_BYTE *)(v8 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v8 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) == 0 )
        break;
      v9 = *(_QWORD *)v8;
    }
  }
  return (unsigned int)*(unsigned __int16 *)(v8 + 68) - 38 > 1;
}
