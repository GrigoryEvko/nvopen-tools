// Function: sub_140CA20
// Address: 0x140ca20
//
__int64 __fastcall sub_140CA20(__int64 a1, __int64 a2, __int64 a3)
{
  if ( !(unsigned __int8)sub_15E4F60(a3) )
    __asm { jmp     rax }
  *(_DWORD *)(a1 + 8) = 1;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  return a1;
}
