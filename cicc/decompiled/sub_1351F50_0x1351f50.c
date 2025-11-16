// Function: sub_1351F50
// Address: 0x1351f50
//
__int64 __fastcall sub_1351F50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  _BYTE *v5; // rax
  __int64 v6; // rdi
  _BYTE *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi

  v4 = sub_16E8CB0(a1, a2, a3);
  v5 = *(_BYTE **)(v4 + 24);
  if ( *(_BYTE **)(v4 + 16) == v5 )
  {
    v4 = sub_16E7EE0(v4, "(", 1);
  }
  else
  {
    *v5 = 40;
    ++*(_QWORD *)(v4 + 24);
  }
  v6 = sub_16E7AF0(v4, 100 * a1 / a2, 100 * a1 % a2);
  v7 = *(_BYTE **)(v6 + 24);
  if ( *(_BYTE **)(v6 + 16) == v7 )
  {
    v6 = sub_16E7EE0(v6, ".", 1);
  }
  else
  {
    *v7 = 46;
    ++*(_QWORD *)(v6 + 24);
  }
  v8 = sub_16E7AF0(v6, 1000 * a1 / a2 % 10, 1000 * a1 / a2 / 10);
  v9 = *(_QWORD *)(v8 + 24);
  v10 = v8;
  if ( (unsigned __int64)(*(_QWORD *)(v8 + 16) - v9) <= 2 )
    return sub_16E7EE0(v8, "%)\n", 3);
  *(_BYTE *)(v9 + 2) = 10;
  *(_WORD *)v9 = 10533;
  *(_QWORD *)(v10 + 24) += 3LL;
  return 10533;
}
