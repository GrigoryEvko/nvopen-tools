// Function: sub_2CDFE70
// Address: 0x2cdfe70
//
__int64 __fastcall sub_2CDFE70(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(*(_QWORD *)(a1 - 32) + 8LL);
  if ( (unsigned int)*(unsigned __int8 *)(v1 + 8) - 17 <= 1 )
    v1 = **(_QWORD **)(v1 + 16);
  return *(_DWORD *)(v1 + 8) >> 8;
}
