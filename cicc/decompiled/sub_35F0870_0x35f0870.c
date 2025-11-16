// Function: sub_35F0870
// Address: 0x35f0870
//
unsigned __int64 __fastcall sub_35F0870(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // rbx
  char v6; // al
  unsigned __int64 result; // rax
  __int64 v8; // rdx
  _DWORD *v9; // rdx
  _QWORD *v10; // rdx
  _DWORD *v11; // rdx
  __int64 v12; // rdx

  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  if ( (v5 & 0x200) != 0 )
  {
    v8 = *(_QWORD *)(a4 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v8) <= 8 )
    {
      sub_CB6200(a4, (unsigned __int8 *)"::cluster", 9u);
    }
    else
    {
      *(_BYTE *)(v8 + 8) = 114;
      *(_QWORD *)v8 = 0x657473756C633A3ALL;
      *(_QWORD *)(a4 + 32) += 9LL;
    }
  }
  v6 = (unsigned __int8)v5 >> 4;
  if ( (unsigned __int8)v5 >> 4 == 2 )
  {
    v11 = *(_DWORD **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v11 <= 3u )
    {
      sub_CB6200(a4, (unsigned __int8 *)".sys", 4u);
    }
    else
    {
      *v11 = 1937339182;
      *(_QWORD *)(a4 + 32) += 4LL;
    }
  }
  else if ( v6 == 3 )
  {
    v10 = *(_QWORD **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v10 <= 7u )
    {
      sub_CB6200(a4, (unsigned __int8 *)".cluster", 8u);
    }
    else
    {
      *v10 = 0x72657473756C632ELL;
      *(_QWORD *)(a4 + 32) += 8LL;
    }
  }
  else if ( v6 == 1 )
  {
    v9 = *(_DWORD **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v9 <= 3u )
    {
      sub_CB6200(a4, (unsigned __int8 *)".cta", 4u);
    }
    else
    {
      *v9 = 1635017518;
      *(_QWORD *)(a4 + 32) += 4LL;
    }
  }
  result = sub_35ED4A0(v5 & 0xF, a4);
  if ( BYTE2(v5) == 11 )
  {
    v12 = *(_QWORD *)(a4 + 32);
    result = *(_QWORD *)(a4 + 24) - v12;
    if ( result <= 4 )
    {
      return sub_CB6200(a4, (unsigned __int8 *)".add.", 5u);
    }
    else
    {
      *(_DWORD *)v12 = 1684300078;
      *(_BYTE *)(v12 + 4) = 46;
      *(_QWORD *)(a4 + 32) += 5LL;
    }
  }
  return result;
}
