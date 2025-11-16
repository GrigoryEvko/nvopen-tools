// Function: sub_35EFA90
// Address: 0x35efa90
//
size_t __fastcall sub_35EFA90(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  size_t result; // rax
  _QWORD *v9; // rdx

  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( !a5 )
    return sub_CB59F0(a4, (int)v6);
  result = strlen((const char *)a5);
  if ( !result )
    return sub_CB59F0(a4, (int)v6);
  if ( result != 7 )
    goto LABEL_16;
  if ( *(_DWORD *)a5 == 1936876918 && *(_WORD *)(a5 + 4) == 28521 && *(_BYTE *)(a5 + 6) == 110 )
    return sub_CB59F0(a4, (int)v6);
  if ( *(_DWORD *)a5 != 1734962273 || *(_WORD *)(a5 + 4) != 25966 || *(_BYTE *)(a5 + 6) != 100 )
LABEL_16:
    BUG();
  if ( (int)v6 > 62 )
  {
    v9 = *(_QWORD **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v9 <= 7u )
    {
      return sub_CB6200(a4, (unsigned __int8 *)".aligned", 8u);
    }
    else
    {
      *v9 = 0x64656E67696C612ELL;
      *(_QWORD *)(a4 + 32) += 8LL;
      return 0x64656E67696C612ELL;
    }
  }
  return result;
}
