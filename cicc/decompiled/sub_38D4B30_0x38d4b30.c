// Function: sub_38D4B30
// Address: 0x38d4b30
//
unsigned __int64 __fastcall sub_38D4B30(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rdx
  __int64 v3; // rcx
  unsigned __int64 result; // rax

  v1 = *(unsigned int *)(a1 + 120);
  if ( !(_DWORD)v1 )
    BUG();
  v2 = *(_QWORD **)(a1 + 272);
  v3 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v1 - 32);
  result = 0;
  if ( v2 != *(_QWORD **)(v3 + 104) )
    return *v2 & 0xFFFFFFFFFFFFFFF8LL;
  return result;
}
