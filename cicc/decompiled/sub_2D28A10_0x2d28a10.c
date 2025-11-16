// Function: sub_2D28A10
// Address: 0x2d28a10
//
__int64 __fastcall sub_2D28A10(__int64 a1)
{
  return *(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16)
       + 8LL * *(unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16 + 12)
       + 4;
}
