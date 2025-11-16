// Function: sub_F80650
// Address: 0xf80650
//
__int64 __fastcall sub_F80650(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // r14
  unsigned int v7; // eax
  unsigned int v8; // r12d
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  char *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rsi
  char *v17; // rcx

  v6 = (char *)a3;
  v7 = sub_F80610((__int64)a1, a2, a3, a4, a5, a6);
  if ( !(_BYTE)v7 )
    return 0;
  v8 = v7;
  if ( sub_DAEB50(*a1, a2, *((_QWORD *)v6 + 5)) )
    return v8;
  if ( !sub_DAEB40(*a1, a2, *((_QWORD *)v6 + 5)) )
    return 0;
  v10 = *((_QWORD *)v6 + 5);
  v11 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 != v10 + 48 )
  {
    if ( !v11 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 <= 0xA && v6 == (char *)(v11 - 24) )
      return v8;
  }
  if ( *(_WORD *)(a2 + 24) != 15 )
    return 0;
  v12 = *(_QWORD *)(a2 - 8);
  v13 = 32LL * (*((_DWORD *)v6 + 1) & 0x7FFFFFF);
  if ( (v6[7] & 0x40) != 0 )
  {
    v14 = (char *)*((_QWORD *)v6 - 1);
    v6 = &v14[v13];
  }
  else
  {
    v14 = &v6[-v13];
  }
  v15 = (v6 - v14) >> 7;
  v16 = (v6 - v14) >> 5;
  if ( v15 <= 0 )
  {
LABEL_24:
    if ( v16 != 2 )
    {
      if ( v16 != 3 )
      {
        if ( v16 == 1 )
        {
LABEL_27:
          v8 = 0;
          if ( v12 != *(_QWORD *)v14 )
            return v8;
LABEL_28:
          LOBYTE(v8) = v14 != v6;
          return v8;
        }
        return 0;
      }
      if ( v12 == *(_QWORD *)v14 )
        goto LABEL_28;
      v14 += 32;
    }
    if ( v12 == *(_QWORD *)v14 )
      goto LABEL_28;
    v14 += 32;
    goto LABEL_27;
  }
  v17 = &v14[128 * v15];
  while ( 1 )
  {
    if ( v12 == *(_QWORD *)v14 )
      goto LABEL_20;
    if ( v12 == *((_QWORD *)v14 + 4) )
    {
      v14 += 32;
LABEL_20:
      LOBYTE(v8) = v6 != v14;
      return v8;
    }
    if ( v12 == *((_QWORD *)v14 + 8) )
    {
      LOBYTE(v8) = v6 != v14 + 64;
      return v8;
    }
    if ( v12 == *((_QWORD *)v14 + 12) )
      break;
    v14 += 128;
    if ( v17 == v14 )
    {
      v16 = (v6 - v14) >> 5;
      goto LABEL_24;
    }
  }
  LOBYTE(v8) = v6 != v14 + 96;
  return v8;
}
