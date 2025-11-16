// Function: sub_CF5E90
// Address: 0xcf5e90
//
__int64 __fastcall sub_CF5E90(__int64 a1, int a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rdx
  __int64 v5; // rdx
  __int64 v6; // rdx
  void *v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdi
  _BYTE *v10; // rax

  v2 = a1;
  if ( (_BYTE)a2 == 2 )
  {
    v7 = *(void **)(a1 + 32);
    if ( *(_QWORD *)(a1 + 24) - (_QWORD)v7 <= 0xBu )
    {
      sub_CB6200(a1, "PartialAlias", 0xCu);
    }
    else
    {
      qmemcpy(v7, "PartialAlias", 12);
      *(_QWORD *)(a1 + 32) += 12LL;
    }
    if ( (a2 & 0x100) != 0 )
    {
      v8 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v8) <= 5 )
      {
        a1 = sub_CB6200(a1, " (off ", 6u);
      }
      else
      {
        *(_DWORD *)v8 = 1718560800;
        *(_WORD *)(v8 + 4) = 8294;
        *(_QWORD *)(a1 + 32) += 6LL;
      }
      v9 = sub_CB59F0(a1, a2 >> 9);
      v10 = *(_BYTE **)(v9 + 32);
      if ( *(_BYTE **)(v9 + 24) == v10 )
      {
        sub_CB6200(v9, (unsigned __int8 *)")", 1u);
      }
      else
      {
        *v10 = 41;
        ++*(_QWORD *)(v9 + 32);
      }
    }
  }
  else if ( (unsigned __int8)a2 > 2u )
  {
    if ( (_BYTE)a2 == 3 )
    {
      v5 = *(_QWORD *)(a1 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v5) <= 8 )
      {
        sub_CB6200(a1, "MustAlias", 9u);
      }
      else
      {
        *(_BYTE *)(v5 + 8) = 115;
        *(_QWORD *)v5 = 0x61696C417473754DLL;
        *(_QWORD *)(a1 + 32) += 9LL;
      }
    }
  }
  else if ( (_BYTE)a2 )
  {
    v3 = *(_QWORD **)(a1 + 32);
    if ( *(_QWORD *)(a1 + 24) - (_QWORD)v3 <= 7u )
    {
      sub_CB6200(a1, "MayAlias", 8u);
    }
    else
    {
      *v3 = 0x7361696C4179614DLL;
      *(_QWORD *)(a1 + 32) += 8LL;
    }
  }
  else
  {
    v6 = *(_QWORD *)(a1 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a1 + 24) - v6) <= 6 )
    {
      sub_CB6200(a1, "NoAlias", 7u);
    }
    else
    {
      *(_DWORD *)v6 = 1816227662;
      *(_WORD *)(v6 + 4) = 24937;
      *(_BYTE *)(v6 + 6) = 115;
      *(_QWORD *)(a1 + 32) += 7LL;
    }
  }
  return v2;
}
