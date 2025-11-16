// Function: sub_19E4DA0
// Address: 0x19e4da0
//
__int64 __fastcall sub_19E4DA0(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // r14
  __int64 i; // rbx
  __int64 v6; // rdi
  __int64 v7; // rax
  _DWORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rax
  _WORD *v11; // rdx
  _BYTE *v12; // rax

  if ( a3 )
    sub_1263B40(a2, "ExpressionTypeAggregateValue, ");
  sub_1930810(a1, a2, 0);
  sub_1263B40(a2, ", intoperands = {");
  v4 = *(unsigned int *)(a1 + 52);
  if ( (_DWORD)v4 )
  {
    for ( i = 0; i != v4; ++i )
    {
      while ( 1 )
      {
        v12 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v12 )
        {
          v6 = sub_16E7EE0(a2, "[", 1u);
        }
        else
        {
          *v12 = 91;
          v6 = a2;
          ++*(_QWORD *)(a2 + 24);
        }
        v7 = sub_16E7A90(v6, i);
        v8 = *(_DWORD **)(v7 + 24);
        v9 = v7;
        if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 3u )
        {
          v9 = sub_16E7EE0(v7, "] = ", 4u);
        }
        else
        {
          *v8 = 540876893;
          *(_QWORD *)(v7 + 24) += 4LL;
        }
        v10 = sub_16E7A90(v9, *(unsigned int *)(*(_QWORD *)(a1 + 56) + 4 * i));
        v11 = *(_WORD **)(v10 + 24);
        if ( *(_QWORD *)(v10 + 16) - (_QWORD)v11 <= 1u )
          break;
        ++i;
        *v11 = 8224;
        *(_QWORD *)(v10 + 24) += 2LL;
        if ( v4 == i )
          return sub_1263B40(a2, "}");
      }
      sub_16E7EE0(v10, "  ", 2u);
    }
  }
  return sub_1263B40(a2, "}");
}
