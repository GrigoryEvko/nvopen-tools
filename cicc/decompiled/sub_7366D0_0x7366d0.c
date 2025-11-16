// Function: sub_7366D0
// Address: 0x7366d0
//
__int64 __fastcall sub_7366D0(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // r9
  __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // cl
  __int64 v7; // r9
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rdx

  if ( a2 == -1 )
    BUG();
  v2 = 776LL * a2;
  v3 = qword_4F04C68[0] + v2;
  if ( (*(_BYTE *)(qword_4F04C68[0] + v2 + 8) & 4) != 0 )
  {
    if ( !a2 || (v4 = v2 - 776, (v5 = v4 + qword_4F04C68[0]) == 0) )
      BUG();
    v6 = *(_BYTE *)(v5 + 4);
    a2 = 1594008481 * (v4 >> 3);
    if ( a2 >= 0 )
    {
      v7 = *(_QWORD *)(v3 + 208);
      while ( 1 )
      {
        if ( *(_BYTE *)(v5 + 4) == 6 && *(_QWORD *)(v5 + 208) == v7 )
        {
          a2 = 1594008481 * ((v5 - qword_4F04C68[0]) >> 3);
          return sub_7365B0(a1, a2);
        }
        a2 = 1594008481 * ((v5 - 776 - qword_4F04C68[0]) >> 3);
        if ( a2 < 0 )
          break;
        v5 -= 776;
      }
      v6 = *(_BYTE *)(v5 - 772);
    }
  }
  else
  {
    v6 = *(_BYTE *)(v3 + 4);
  }
  if ( v6 == 7 )
  {
    sub_735D50(a1, a2);
    v11 = *(_QWORD *)(a1 + 40);
    result = *(_QWORD *)(v11 + 104);
    if ( result )
    {
      do
      {
        v12 = result;
        result = *(_QWORD *)(result + 112);
      }
      while ( result );
      *(_QWORD *)(v12 + 112) = a1;
    }
    else
    {
      *(_QWORD *)(v11 + 104) = a1;
    }
  }
  else if ( v6 == 5 )
  {
    v9 = 776LL * a2;
    v10 = *(_QWORD *)(*(_QWORD *)(qword_4F04C68[0] + v9 + 224) + 128LL);
    sub_72EE40(a1, 6u, v10);
    result = *(_QWORD *)(qword_4F04C68[0] + v9 + 24);
    if ( *(_QWORD *)(v10 + 104) )
      *(_QWORD *)(*(_QWORD *)(result + 32) + 112LL) = a1;
    else
      *(_QWORD *)(v10 + 104) = a1;
    *(_QWORD *)(result + 32) = a1;
  }
  else
  {
    return sub_7365B0(a1, a2);
  }
  return result;
}
