// Function: sub_2E10F30
// Address: 0x2e10f30
//
__int64 __fastcall sub_2E10F30(int a1)
{
  int v1; // xmm0_4
  __int64 result; // rax

  v1 = 0;
  if ( (unsigned int)(a1 - 1) <= 0x3FFFFFFE )
    v1 = unk_44D0BE0;
  result = sub_22077B0(0x78u);
  if ( result )
  {
    *(_DWORD *)(result + 112) = a1;
    *(_QWORD *)result = result + 16;
    *(_QWORD *)(result + 8) = 0x200000000LL;
    *(_QWORD *)(result + 64) = result + 80;
    *(_QWORD *)(result + 72) = 0x200000000LL;
    *(_QWORD *)(result + 96) = 0;
    *(_QWORD *)(result + 104) = 0;
    *(_DWORD *)(result + 116) = v1;
  }
  return result;
}
