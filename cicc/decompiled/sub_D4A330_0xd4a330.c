// Function: sub_D4A330
// Address: 0xd4a330
//
__int64 __fastcall sub_D4A330(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned int *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax

  v1 = 0x8000000000041LL;
  v2 = *(_QWORD *)(a1 + 32);
  v3 = *(_QWORD *)(*(_QWORD *)v2 + 56LL);
  v4 = *(_QWORD *)v2 + 48LL;
  if ( v3 == v4 )
    return 0;
  while ( 1 )
  {
    if ( !v3 )
      BUG();
    v5 = v3 - 24;
    if ( (unsigned __int8)(*(_BYTE *)(v3 - 24) - 34) <= 0x33u
      && _bittest64(&v1, (unsigned int)*(unsigned __int8 *)(v3 - 24) - 34)
      && ((unsigned __int8)sub_A73ED0((_QWORD *)(v3 + 48), 6) || (unsigned __int8)sub_B49560(v3 - 24, 6)) )
    {
      break;
    }
    v3 = *(_QWORD *)(v3 + 8);
    if ( v4 == v3 )
      return 0;
  }
  if ( *(char *)(v3 - 17) >= 0 )
    return 0;
  v7 = sub_BD2BC0(v3 - 24);
  v10 = v7 + v9;
  if ( *(char *)(v3 - 17) < 0 )
    v10 -= sub_BD2BC0(v3 - 24);
  v11 = v10 >> 4;
  if ( !(_DWORD)v11 )
    return 0;
  v12 = 0;
  v13 = 16LL * (unsigned int)v11;
  while ( 1 )
  {
    v14 = 0;
    if ( *(char *)(v3 - 17) < 0 )
      v14 = sub_BD2BC0(v3 - 24);
    v15 = (unsigned int *)(v12 + v14);
    if ( *(_DWORD *)(*(_QWORD *)v15 + 8LL) == 9 )
      break;
    v12 += 16;
    if ( v12 == v13 )
      return 0;
  }
  v16 = *(_DWORD *)(v3 - 20) & 0x7FFFFFF;
  v17 = *(_QWORD *)(v3 - 24 + 32 * (v15[2] - v16));
  if ( !v17 || (unsigned __int8)sub_B19060(a1 + 56, *(_QWORD *)(v17 + 40), v16, v8) )
    return 0;
  return v5;
}
