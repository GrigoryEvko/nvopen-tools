// Function: sub_1552500
// Address: 0x1552500
//
__int64 __fastcall sub_1552500(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r15
  __int64 v5; // r13
  __int64 v6; // rdi
  void *v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdi
  _DWORD *v10; // rdx
  unsigned int v11; // r14d
  int v12; // r13d
  __int64 v13; // rdx
  __int64 v14; // rdi
  _WORD *v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // r13

  result = a1[39];
  if ( a1[38] != result && *(_QWORD *)(result - 32) == a2 )
  {
    result = sub_1263B40(*a1, "\n; uselistorder directives\n");
    while ( 1 )
    {
      v4 = a1[39];
      if ( v4 == a1[38] || *(_QWORD *)(v4 - 32) != a2 )
        break;
      v5 = *(_QWORD *)(a1[4] + 8);
      if ( v5 )
        sub_1263B40(*a1, "  ");
      v6 = *a1;
      v7 = *(void **)(*a1 + 24);
      if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v7 <= 0xBu )
      {
        sub_16E7EE0(v6, "uselistorder", 12);
      }
      else
      {
        qmemcpy(v7, "uselistorder", 12);
        *(_QWORD *)(v6 + 24) += 12LL;
      }
      v8 = *a1;
      if ( v5 || (v19 = *(_QWORD *)(v4 - 40), *(_BYTE *)(v19 + 16) != 18) )
      {
        sub_1263B40(v8, " ");
        sub_15520E0(a1, *(__int64 **)(v4 - 40), 1);
      }
      else
      {
        sub_1263B40(v8, "_bb ");
        sub_15520E0(a1, *(__int64 **)(v19 + 56), 0);
        sub_1263B40(*a1, ", ");
        sub_15520E0(a1, (__int64 *)v19, 0);
      }
      v9 = *a1;
      v10 = *(_DWORD **)(*a1 + 24);
      if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v10 <= 3u )
      {
        sub_16E7EE0(v9, ", { ", 4);
      }
      else
      {
        *v10 = 544940076;
        *(_QWORD *)(v9 + 24) += 4LL;
      }
      v11 = 1;
      sub_16E7A90(*a1, **(unsigned int **)(v4 - 24));
      v12 = (__int64)(*(_QWORD *)(v4 - 16) - *(_QWORD *)(v4 - 24)) >> 2;
      if ( v12 != 1 )
      {
        do
        {
          v14 = *a1;
          v15 = *(_WORD **)(*a1 + 24);
          if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v15 > 1u )
          {
            *v15 = 8236;
            *(_QWORD *)(v14 + 24) += 2LL;
          }
          else
          {
            v14 = sub_16E7EE0(v14, ", ", 2);
          }
          v13 = v11++;
          sub_16E7A90(v14, *(unsigned int *)(*(_QWORD *)(v4 - 24) + 4 * v13));
        }
        while ( v12 != v11 );
      }
      v16 = *a1;
      v17 = *(_QWORD *)(*a1 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(*a1 + 16) - v17) <= 2 )
      {
        sub_16E7EE0(v16, " }\n", 3);
      }
      else
      {
        *(_BYTE *)(v17 + 2) = 10;
        *(_WORD *)v17 = 32032;
        *(_QWORD *)(v16 + 24) += 3LL;
      }
      result = a1[39];
      a1[39] = result - 40;
      v18 = *(_QWORD *)(result - 24);
      if ( v18 )
        result = j_j___libc_free_0(v18, *(_QWORD *)(result - 8) - v18);
    }
  }
  return result;
}
