// Function: sub_1BE2B90
// Address: 0x1be2b90
//
__int64 __fastcall sub_1BE2B90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r15
  __int64 v6; // rdx
  __int64 result; // rax
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 i; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v4) <= 2 )
  {
    v5 = sub_16E7EE0(a2, " +\n", 3u);
  }
  else
  {
    *(_BYTE *)(v4 + 2) = 10;
    v5 = a2;
    *(_WORD *)v4 = 11040;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_16E2CE0(a3, v5);
  v6 = *(_QWORD *)(v5 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) <= 8 )
  {
    sub_16E7EE0(v5, "\"WIDEN\\l\"", 9u);
  }
  else
  {
    *(_BYTE *)(v6 + 8) = 34;
    *(_QWORD *)v6 = 0x6C5C4E4544495722LL;
    *(_QWORD *)(v5 + 24) += 9LL;
  }
  result = *(_QWORD *)(a1 + 48);
  v8 = *(_QWORD *)(a1 + 40);
  for ( i = result; i != v8; v8 = *(_QWORD *)(v8 + 8) )
  {
    while ( 1 )
    {
      v12 = v8 - 24;
      v13 = *(_QWORD *)(a2 + 24);
      if ( !v8 )
        v12 = 0;
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v13) > 2 )
      {
        *(_BYTE *)(v13 + 2) = 10;
        v9 = a2;
        *(_WORD *)v13 = 11040;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      else
      {
        v9 = sub_16E7EE0(a2, " +\n", 3u);
      }
      sub_16E2CE0(a3, v9);
      v10 = *(_QWORD *)(v9 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v9 + 16) - v10) <= 2 )
      {
        v9 = sub_16E7EE0(v9, "\"  ", 3u);
      }
      else
      {
        *(_BYTE *)(v10 + 2) = 32;
        *(_WORD *)v10 = 8226;
        *(_QWORD *)(v9 + 24) += 3LL;
      }
      sub_1BE27E0(v9, v12);
      v11 = *(_QWORD *)(v9 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v9 + 16) - v11) <= 2 )
        break;
      result = 27740;
      *(_BYTE *)(v11 + 2) = 34;
      *(_WORD *)v11 = 27740;
      *(_QWORD *)(v9 + 24) += 3LL;
      v8 = *(_QWORD *)(v8 + 8);
      if ( i == v8 )
        return result;
    }
    result = sub_16E7EE0(v9, "\\l\"", 3u);
  }
  return result;
}
