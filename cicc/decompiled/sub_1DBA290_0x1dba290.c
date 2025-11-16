// Function: sub_1DBA290
// Address: 0x1dba290
//
__int64 __fastcall sub_1DBA290(int a1)
{
  int v1; // xmm0_4
  __int64 result; // rax

  v1 = 0;
  if ( a1 > 0 )
    v1 = unk_4530D80;
  result = sub_22077B0(120);
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
