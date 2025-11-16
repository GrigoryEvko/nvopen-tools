// Function: sub_15F8650
// Address: 0x15f8650
//
unsigned __int64 __fastcall sub_15F8650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax

  v7 = sub_157E9C0(a2);
  v8 = sub_1643270(v7);
  result = sub_15F1F50(a1, v8, 2, a1 - 72, 3, a5);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v10 = *(_QWORD *)(a1 - 16);
    result = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)result = v10;
    if ( v10 )
    {
      result |= *(_QWORD *)(v10 + 16) & 3LL;
      *(_QWORD *)(v10 + 16) = result;
    }
  }
  *(_QWORD *)(a1 - 24) = a2;
  if ( a2 )
  {
    v11 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 16) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = (a1 - 16) | *(_QWORD *)(v11 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a2 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    result = a1 - 24;
    *(_QWORD *)(a2 + 8) = a1 - 24;
  }
  if ( *(_QWORD *)(a1 - 48) )
  {
    v12 = *(_QWORD *)(a1 - 40);
    result = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)result = v12;
    if ( v12 )
    {
      result |= *(_QWORD *)(v12 + 16) & 3LL;
      *(_QWORD *)(v12 + 16) = result;
    }
  }
  *(_QWORD *)(a1 - 48) = a3;
  if ( a3 )
  {
    v13 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 40) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = (a1 - 40) | *(_QWORD *)(v13 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (a3 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    result = a1 - 48;
    *(_QWORD *)(a3 + 8) = a1 - 48;
  }
  if ( *(_QWORD *)(a1 - 72) )
  {
    v14 = *(_QWORD *)(a1 - 64);
    result = *(_QWORD *)(a1 - 56) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)result = v14;
    if ( v14 )
    {
      result |= *(_QWORD *)(v14 + 16) & 3LL;
      *(_QWORD *)(v14 + 16) = result;
    }
  }
  *(_QWORD *)(a1 - 72) = a4;
  if ( a4 )
  {
    v15 = *(_QWORD *)(a4 + 8);
    *(_QWORD *)(a1 - 64) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = (a1 - 64) | *(_QWORD *)(v15 + 16) & 3LL;
    v16 = *(_QWORD *)(a1 - 56);
    *(_QWORD *)(a4 + 8) = a1 - 72;
    result = (a4 + 8) | v16 & 3;
    *(_QWORD *)(a1 - 56) = result;
  }
  return result;
}
