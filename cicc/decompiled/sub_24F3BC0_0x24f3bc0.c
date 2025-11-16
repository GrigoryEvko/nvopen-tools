// Function: sub_24F3BC0
// Address: 0x24f3bc0
//
__int64 __fastcall sub_24F3BC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  _QWORD *v3; // rax
  __int64 result; // rax

  v2 = *((_QWORD *)sub_BD3990(*(unsigned __int8 **)(a1 + 32 * (2LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), a2) + 3);
  v3 = *(_QWORD **)(v2 + 16);
  if ( *(_BYTE *)(*v3 + 8LL) != 14 )
    sub_C64ED0("llvm.coro.suspend.async resume function projection function must return a ptr type", 1u);
  if ( *(_DWORD *)(v2 + 12) != 2 || (result = v3[1], *(_BYTE *)(result + 8) != 14) )
    sub_C64ED0("llvm.coro.suspend.async resume function projection function must take one ptr type as parameter", 1u);
  return result;
}
