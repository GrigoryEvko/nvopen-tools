// Function: sub_1B91F40
// Address: 0x1b91f40
//
__int64 __fastcall sub_1B91F40(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r13
  __int64 v7; // rbx
  __int64 *v8; // rax

  result = sub_157F280(*(_QWORD *)(a1 + 192));
  if ( v2 != result )
  {
    v6 = v2;
    v7 = result;
    do
    {
      if ( (*(_DWORD *)(v7 + 20) & 0xFFFFFFF) == 1 )
      {
        v8 = (__int64 *)(v7 - 24);
        if ( (*(_BYTE *)(v7 + 23) & 0x40) != 0 )
          v8 = *(__int64 **)(v7 - 8);
        sub_1704F80(v7, *v8, *(_QWORD *)(a1 + 184), v3, v4, v5);
      }
      result = *(_QWORD *)(v7 + 32);
      if ( !result )
        BUG();
      v7 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v7 = result - 24;
    }
    while ( v6 != v7 );
  }
  return result;
}
