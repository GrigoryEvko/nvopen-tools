// Function: sub_309F080
// Address: 0x309f080
//
__int64 __fastcall sub_309F080(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // rax
  __int64 v5; // rdi
  _BYTE *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdi

  v3 = (__int64)sub_CB72A0();
  v4 = *(_BYTE **)(v3 + 32);
  if ( *(_BYTE **)(v3 + 24) == v4 )
  {
    v3 = sub_CB6200(v3, (unsigned __int8 *)"(", 1u);
  }
  else
  {
    *v4 = 40;
    ++*(_QWORD *)(v3 + 32);
  }
  v5 = sub_CB5A30(v3, 100 * a1 / a2);
  v6 = *(_BYTE **)(v5 + 32);
  if ( *(_BYTE **)(v5 + 24) == v6 )
  {
    v5 = sub_CB6200(v5, (unsigned __int8 *)".", 1u);
  }
  else
  {
    *v6 = 46;
    ++*(_QWORD *)(v5 + 32);
  }
  v7 = sub_CB5A30(v5, 1000 * a1 / a2 % 10);
  v8 = *(_QWORD *)(v7 + 32);
  v9 = v7;
  if ( (unsigned __int64)(*(_QWORD *)(v7 + 24) - v8) <= 2 )
    return sub_CB6200(v7, "%)\n", 3u);
  *(_BYTE *)(v8 + 2) = 10;
  *(_WORD *)v8 = 10533;
  *(_QWORD *)(v9 + 32) += 3LL;
  return 10533;
}
