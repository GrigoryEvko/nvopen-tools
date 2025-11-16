// Function: sub_2D28A30
// Address: 0x2d28a30
//
__int64 __fastcall sub_2D28A30(__int64 a1)
{
  return *(_QWORD *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16)
       + 4LL * *(unsigned int *)(*(_QWORD *)(a1 + 8) + 16LL * *(unsigned int *)(a1 + 16) - 16 + 12)
       + 128;
}
