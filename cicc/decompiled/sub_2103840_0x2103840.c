// Function: sub_2103840
// Address: 0x2103840
//
__int64 __fastcall sub_2103840(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  int *v4; // rdx
  int v5; // ecx
  int v6; // edx

  result = *(_QWORD *)(a1 + 392) + 176LL * a3;
  v4 = (int *)(*(_QWORD *)(a1 + 384) + 216LL * a3);
  v5 = *(_DWORD *)(a1 + 256);
  if ( v5 != *(_DWORD *)(result + 168)
    || a2 != *(_QWORD *)(result + 8)
    || v4 != *(int **)result
    || *(_DWORD *)(result + 164) != *v4 )
  {
    *(_QWORD *)(result + 8) = a2;
    *(_QWORD *)result = v4;
    *(_DWORD *)(result + 120) = 0;
    *(_WORD *)(result + 160) = 0;
    v6 = *v4;
    *(_DWORD *)(result + 168) = v5;
    *(_DWORD *)(result + 164) = v6;
  }
  return result;
}
