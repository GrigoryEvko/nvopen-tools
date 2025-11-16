// Function: sub_165C060
// Address: 0x165c060
//
__int64 __fastcall sub_165C060(_QWORD *a1, int a2)
{
  unsigned __int64 v2; // rbx
  char v3; // dl
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r13
  unsigned int v9; // r12d
  __int64 v10; // r15
  __int64 v11; // rax
  bool v12; // al
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  __int64 v17; // r15
  __int64 v18; // r13
  __int64 v19; // r15
  __int64 v20; // rax
  bool v21; // al

  v2 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v3 = *(_BYTE *)(v2 + 23);
  if ( (*a1 & 4) != 0 )
  {
    if ( v3 < 0 )
    {
      v4 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
      v6 = v4 + v5;
      if ( *(char *)(v2 + 23) >= 0 )
        v7 = v6 >> 4;
      else
        LODWORD(v7) = (v6 - sub_1648A40(v2)) >> 4;
      v8 = 0;
      v9 = 0;
      v10 = 16LL * (unsigned int)v7;
      if ( (_DWORD)v7 )
      {
        do
        {
          v11 = 0;
          if ( *(char *)(v2 + 23) < 0 )
            v11 = sub_1648A40(v2);
          v12 = a2 == *(_DWORD *)(*(_QWORD *)(v11 + v8) + 8LL);
          v8 += 16;
          v9 += v12;
        }
        while ( v8 != v10 );
        return v9;
      }
    }
    return 0;
  }
  if ( v3 >= 0 )
    return 0;
  v14 = sub_1648A40(*a1 & 0xFFFFFFFFFFFFFFF8LL);
  v16 = v14 + v15;
  if ( *(char *)(v2 + 23) < 0 )
    v16 -= sub_1648A40(v2);
  v17 = v16 >> 4;
  if ( !(_DWORD)v17 )
    return 0;
  v18 = 0;
  v9 = 0;
  v19 = 16LL * (unsigned int)v17;
  do
  {
    v20 = 0;
    if ( *(char *)(v2 + 23) < 0 )
      v20 = sub_1648A40(v2);
    v21 = a2 == *(_DWORD *)(*(_QWORD *)(v20 + v18) + 8LL);
    v18 += 16;
    v9 += v21;
  }
  while ( v19 != v18 );
  return v9;
}
