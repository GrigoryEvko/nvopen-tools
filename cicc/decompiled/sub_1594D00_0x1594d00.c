// Function: sub_1594D00
// Address: 0x1594d00
//
unsigned __int64 __fastcall sub_1594D00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  unsigned __int64 result; // rax
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax

  v4 = sub_15E0530(a2);
  v5 = sub_16471D0(v4, 0);
  sub_1648CB0(a1, v5, 4);
  result = *(_DWORD *)(a1 + 20) & 0xF0000000 | 2;
  v7 = *(_QWORD *)(a1 - 48) == 0;
  *(_DWORD *)(a1 + 20) = result;
  if ( !v7 )
  {
    v8 = *(_QWORD *)(a1 - 40);
    result = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)result = v8;
    if ( v8 )
    {
      result |= *(_QWORD *)(v8 + 16) & 3LL;
      *(_QWORD *)(v8 + 16) = result;
    }
  }
  *(_QWORD *)(a1 - 48) = a2;
  if ( a2 )
  {
    v9 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 40) = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = (a1 - 40) | *(_QWORD *)(v9 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (a2 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    result = a1 - 48;
    *(_QWORD *)(a2 + 8) = a1 - 48;
  }
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
  *(_QWORD *)(a1 - 24) = a3;
  if ( a3 )
  {
    v11 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 16) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = (a1 - 16) | *(_QWORD *)(v11 + 16) & 3LL;
    result = (a3 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a1 - 24 + 16) = result;
    *(_QWORD *)(a3 + 8) = a1 - 24;
  }
  ++*(_WORD *)(a3 + 18);
  return result;
}
