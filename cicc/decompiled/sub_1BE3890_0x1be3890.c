// Function: sub_1BE3890
// Address: 0x1be3890
//
__int64 __fastcall sub_1BE3890(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rdx
  _WORD *v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rdi
  __int64 v11; // rdx

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
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v6) <= 6 )
  {
    v5 = sub_16E7EE0(v5, "\"WIDEN ", 7u);
  }
  else
  {
    *(_DWORD *)v6 = 1145657122;
    *(_WORD *)(v6 + 4) = 20037;
    *(_BYTE *)(v6 + 6) = 32;
    *(_QWORD *)(v5 + 24) += 7LL;
  }
  sub_1BE27E0(v5, *(_QWORD *)(a1 + 40));
  if ( *(_QWORD *)(a1 + 48) )
  {
    v7 = *(_WORD **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v7 <= 1u )
    {
      sub_16E7EE0(a2, ", ", 2u);
      v8 = *(_QWORD *)(a2 + 24);
    }
    else
    {
      *v7 = 8236;
      v8 = *(_QWORD *)(a2 + 24) + 2LL;
      *(_QWORD *)(a2 + 24) = v8;
    }
    v9 = **(_QWORD **)(*(_QWORD *)(a1 + 48) + 40LL);
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v8) <= 2 )
    {
      v10 = sub_16E7EE0(a2, "%vp", 3u);
    }
    else
    {
      *(_BYTE *)(v8 + 2) = 112;
      v10 = a2;
      *(_WORD *)v8 = 30245;
      *(_QWORD *)(a2 + 24) += 3LL;
    }
    sub_16E7AB0(v10, (unsigned __int16)v9);
  }
  v11 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v11) <= 2 )
    return sub_16E7EE0(a2, "\\l\"", 3u);
  *(_BYTE *)(v11 + 2) = 34;
  *(_WORD *)v11 = 27740;
  *(_QWORD *)(a2 + 24) += 3LL;
  return 27740;
}
