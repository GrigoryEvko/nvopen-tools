// Function: sub_14AB850
// Address: 0x14ab850
//
bool __fastcall sub_14AB850(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v4; // al
  __int64 v6; // rcx
  char v7; // dl
  __int64 v9; // rbx
  char v10; // dl
  __int64 v11; // rdi
  char v12; // si
  unsigned int v13; // r14d
  int v14; // r13d
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r12
  char v19; // al
  __int64 v20; // r12

  v4 = *((_BYTE *)a1 + 16);
  if ( v4 > 0x17u )
  {
    v6 = *a1;
    v7 = *(_BYTE *)(*a1 + 8);
    if ( v7 == 16 )
      v7 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
    if ( (unsigned __int8)(v7 - 1) > 5u && v4 != 76 )
      return 0;
    goto LABEL_6;
  }
  if ( v4 == 5 )
  {
    v6 = *a1;
    v10 = *(_BYTE *)(*a1 + 8);
    v11 = *a1;
    v12 = v10;
    if ( v10 == 16 )
      v12 = *(_BYTE *)(**(_QWORD **)(v6 + 16) + 8LL);
    if ( (unsigned __int8)(v12 - 1) <= 5u || *((_WORD *)a1 + 9) == 52 )
    {
LABEL_6:
      if ( (*((_BYTE *)a1 + 17) & 4) != 0 )
        return 1;
      v10 = *(_BYTE *)(v6 + 8);
      v11 = v6;
    }
LABEL_19:
    if ( v10 == 16 && v4 <= 0x10u )
    {
      v13 = 0;
      v14 = *(_QWORD *)(v11 + 32);
      if ( !v14 )
        return 1;
      while ( 1 )
      {
        v15 = sub_15A0A60(a1, v13);
        v18 = v15;
        if ( !v15 )
          break;
        v19 = *(_BYTE *)(v15 + 16);
        if ( v19 != 9 )
        {
          if ( v19 != 14 )
            break;
          v20 = *(_QWORD *)(v18 + 32) == sub_16982C0(a1, v13, v16, v17) ? *(_QWORD *)(v18 + 40) + 8LL : v18 + 32;
          if ( (*(_BYTE *)(v20 + 18) & 7) == 1 )
            break;
        }
        if ( ++v13 == v14 )
          return 1;
      }
    }
    return 0;
  }
  if ( v4 != 14 )
  {
    v11 = *a1;
    v10 = *(_BYTE *)(v11 + 8);
    goto LABEL_19;
  }
  if ( a1[4] == sub_16982C0(a1, a2, a3, a4) )
    v9 = a1[5] + 8;
  else
    v9 = (__int64)(a1 + 4);
  return (*(_BYTE *)(v9 + 18) & 7) != 1;
}
