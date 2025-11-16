// Function: sub_DDAE80
// Address: 0xddae80
//
__int64 __fastcall sub_DDAE80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = sub_DCD020(*(__int64 **)a1, 38, a2, a3);
  if ( !(_BYTE)result )
    return sub_DDA790(
             *(__int64 **)a1,
             0x26u,
             a2,
             a3,
             **(_QWORD **)(a1 + 8),
             **(_QWORD **)(a1 + 16),
             **(_DWORD **)(a1 + 24) + 1);
  return result;
}
