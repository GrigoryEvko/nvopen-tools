// Function: sub_30ADAE0
// Address: 0x30adae0
//
__int64 __fastcall sub_30ADAE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  signed __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx

  while ( 1 )
  {
    v5 = *(unsigned int *)(a1 + 16);
    if ( !(_DWORD)v5 )
      return 1;
    if ( *(_QWORD *)a1 <= 1u )
      break;
    if ( !(unsigned __int8)sub_30AD010((__int64 *)a1) )
      return 1;
  }
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 144 * v5;
  v9 = v7 + v8;
  v10 = 0x8E38E38E38E38E39LL * (v8 >> 4);
  if ( v10 >> 2 )
  {
    v11 = v7 + 576 * (v10 >> 2);
    while ( !*(_DWORD *)(v7 + 8) || *(_WORD *)(*(_QWORD *)v7 + 8LL) || **(__int64 **)v7 >= 0 )
    {
      if ( *(_DWORD *)(v7 + 152) )
      {
        v12 = *(_QWORD *)(v7 + 144);
        if ( !*(_WORD *)(v12 + 8) && *(__int64 *)v12 < 0 )
        {
          LOBYTE(a5) = v9 == v7 + 144;
          return a5;
        }
      }
      if ( *(_DWORD *)(v7 + 296) )
      {
        v13 = *(_QWORD *)(v7 + 288);
        if ( !*(_WORD *)(v13 + 8) && *(__int64 *)v13 < 0 )
        {
          LOBYTE(a5) = v9 == v7 + 288;
          return a5;
        }
      }
      a5 = *(_DWORD *)(v7 + 440);
      if ( a5 )
      {
        v14 = *(_QWORD *)(v7 + 432);
        if ( !*(_WORD *)(v14 + 8) && *(__int64 *)v14 < 0 )
        {
          LOBYTE(a5) = v9 == v7 + 432;
          return a5;
        }
      }
      v7 += 576;
      if ( v7 == v11 )
      {
        v10 = 0x8E38E38E38E38E39LL * ((v9 - v7) >> 4);
        goto LABEL_23;
      }
    }
    goto LABEL_38;
  }
LABEL_23:
  if ( v10 != 2 )
  {
    if ( v10 != 3 )
    {
      if ( v10 != 1 )
        return 1;
      goto LABEL_26;
    }
    if ( *(_DWORD *)(v7 + 8) && !*(_WORD *)(*(_QWORD *)v7 + 8LL) && **(__int64 **)v7 < 0 )
    {
LABEL_38:
      LOBYTE(a5) = v7 == v9;
      return a5;
    }
    v7 += 144;
  }
  if ( *(_DWORD *)(v7 + 8) && !*(_WORD *)(*(_QWORD *)v7 + 8LL) && **(__int64 **)v7 < 0 )
    goto LABEL_29;
  v7 += 144;
LABEL_26:
  a5 = 1;
  if ( *(_DWORD *)(v7 + 8) && !*(_WORD *)(*(_QWORD *)v7 + 8LL) && **(__int64 **)v7 < 0 )
LABEL_29:
    LOBYTE(a5) = v9 == v7;
  return a5;
}
