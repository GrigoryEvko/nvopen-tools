// Function: sub_24F3850
// Address: 0x24f3850
//
__int64 __fastcall sub_24F3850(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rsi
  __int64 v6; // r12
  unsigned __int8 *v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  unsigned __int8 *v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 result; // rax
  __int64 v14; // r13
  char v15; // al

  v2 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( **(_BYTE **)(a1 - 32 * v2) != 17 )
    sub_C64ED0("size argument to coro.id.retcon.* must be constant", 1u);
  if ( **(_BYTE **)(a1 + 32 * (1 - v2)) != 17 )
    sub_C64ED0("alignment argument to coro.id.retcon.* must be constant", 1u);
  v3 = sub_BD3990(*(unsigned __int8 **)(a1 + 32 * (3 - v2)), a2);
  if ( *v3 )
    sub_C64ED0("llvm.coro.id.retcon.* prototype not a Function", 1u);
  v4 = *(_QWORD *)(a1 - 32);
  if ( !v4 || *(_BYTE *)v4 || (v5 = *(_QWORD *)(a1 + 80), *(_QWORD *)(v4 + 24) != v5) )
    BUG();
  v6 = *((_QWORD *)v3 + 3);
  if ( *(_DWORD *)(v4 + 36) == 50 )
  {
    v14 = **(_QWORD **)(v6 + 16);
    v15 = *(_BYTE *)(v14 + 8);
    if ( v15 != 14
      && (v15 != 15
       || (*(_BYTE *)(v14 + 9) & 1) == 0
       || !*(_DWORD *)(v14 + 12)
       || *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL) != 14) )
    {
      sub_C64ED0("llvm.coro.id.retcon prototype must return pointer as first result", 1u);
    }
    if ( v14 != **(_QWORD **)(*(_QWORD *)(sub_B43CB0(a1) + 24) + 16LL) )
      sub_C64ED0("llvm.coro.id.retcon prototype return type must be same ascurrent function return type", 1u);
  }
  if ( *(_DWORD *)(v6 + 12) == 1 || *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL) + 8LL) != 14 )
    sub_C64ED0("llvm.coro.id.retcon.* prototype must take pointer as its first parameter", 1u);
  v7 = sub_BD3990(*(unsigned __int8 **)(a1 + 32 * (4LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), v5);
  if ( *v7 )
    sub_C64ED0("llvm.coro.* allocator not a Function", 1u);
  v8 = *((_QWORD *)v7 + 3);
  v9 = *(_QWORD **)(v8 + 16);
  if ( *(_BYTE *)(*v9 + 8LL) != 14 )
    sub_C64ED0("llvm.coro.* allocator must return a pointer", 1u);
  if ( *(_DWORD *)(v8 + 12) != 2 || *(_BYTE *)(v9[1] + 8LL) != 12 )
    sub_C64ED0("llvm.coro.* allocator must take integer as only param", 1u);
  v10 = sub_BD3990(*(unsigned __int8 **)(a1 + 32 * (5LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), v5);
  if ( *v10 )
    sub_C64ED0("llvm.coro.* deallocator not a Function", 1u);
  v11 = *((_QWORD *)v10 + 3);
  v12 = *(_QWORD **)(v11 + 16);
  if ( *(_BYTE *)(*v12 + 8LL) != 7 )
    sub_C64ED0("llvm.coro.* deallocator must return void", 1u);
  if ( *(_DWORD *)(v11 + 12) != 2 || (result = v12[1], *(_BYTE *)(result + 8) != 14) )
    sub_C64ED0("llvm.coro.* deallocator must take pointer as only param", 1u);
  return result;
}
