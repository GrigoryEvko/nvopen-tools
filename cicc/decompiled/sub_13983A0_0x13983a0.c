// Function: sub_13983A0
// Address: 0x13983a0
//
__int64 __fastcall sub_13983A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rsi
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 result; // rax

  v2 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = v4 + 32;
  if ( *(_QWORD *)(v4 + 16) != v2 )
  {
    do
    {
      v4 = v5;
      v5 += 32;
    }
    while ( *(_QWORD *)(v5 - 16) != v2 );
  }
  --*(_DWORD *)(*(_QWORD *)(v4 + 24) + 32LL);
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(_QWORD *)(v4 + 16);
  v8 = *(_QWORD *)(v6 - 16);
  if ( v7 != v8 )
  {
    if ( v7 != 0 && v7 != -8 && v7 != -16 )
    {
      sub_1649B30(v4);
      v8 = *(_QWORD *)(v6 - 16);
    }
    *(_QWORD *)(v4 + 16) = v8;
    if ( v8 != 0 && v8 != -8 && v8 != -16 )
      sub_1649AC0(v4, *(_QWORD *)(v6 - 32) & 0xFFFFFFFFFFFFFFF8LL);
  }
  *(_QWORD *)(v4 + 24) = *(_QWORD *)(v6 - 8);
  v9 = *(_QWORD *)(a1 + 16);
  v10 = v9 - 32;
  *(_QWORD *)(a1 + 16) = v9 - 32;
  result = *(_QWORD *)(v9 - 16);
  if ( result != -8 && result != 0 && result != -16 )
    return sub_1649B30(v10);
  return result;
}
