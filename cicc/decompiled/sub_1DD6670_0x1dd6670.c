// Function: sub_1DD6670
// Address: 0x1dd6670
//
bool __fastcall sub_1DD6670(__int64 a1, __int16 a2, int a3)
{
  __int64 v3; // r9
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // rcx

  v3 = *(_QWORD *)(a1 + 160);
  v4 = *(_QWORD *)(a1 + 152);
  v5 = (v3 - v4) >> 5;
  v6 = (v3 - v4) >> 3;
  if ( v5 <= 0 )
  {
LABEL_11:
    if ( v6 != 2 )
    {
      if ( v6 != 3 )
      {
        if ( v6 != 1 )
          return 0;
        goto LABEL_19;
      }
      if ( a2 == *(_WORD *)v4 )
        goto LABEL_8;
      v4 += 8;
    }
    if ( a2 == *(_WORD *)v4 )
      goto LABEL_8;
    v4 += 8;
LABEL_19:
    if ( a2 != *(_WORD *)v4 )
      return 0;
    goto LABEL_8;
  }
  v7 = v4 + 32 * v5;
  while ( a2 != *(_WORD *)v4 )
  {
    if ( a2 == *(_WORD *)(v4 + 8) )
    {
      v4 += 8;
      break;
    }
    if ( a2 == *(_WORD *)(v4 + 16) )
    {
      v4 += 16;
      break;
    }
    if ( a2 == *(_WORD *)(v4 + 24) )
    {
      v4 += 24;
      break;
    }
    v4 += 32;
    if ( v4 == v7 )
    {
      v6 = (v3 - v4) >> 3;
      goto LABEL_11;
    }
  }
LABEL_8:
  if ( v3 != v4 )
    return (*(_DWORD *)(v4 + 4) & a3) != 0;
  return 0;
}
