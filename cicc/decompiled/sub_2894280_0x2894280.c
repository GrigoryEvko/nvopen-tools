// Function: sub_2894280
// Address: 0x2894280
//
__int64 __fastcall sub_2894280(__int64 a1)
{
  unsigned int v2; // edx

  if ( *(_BYTE *)a1 == 92
    && (v2 = *(_DWORD *)(a1 + 80), *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL) + 32LL) == v2) )
  {
    return sub_B4EE20(*(int **)(a1 + 72), v2, v2);
  }
  else
  {
    return 0;
  }
}
