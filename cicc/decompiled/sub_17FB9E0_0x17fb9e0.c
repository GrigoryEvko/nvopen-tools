// Function: sub_17FB9E0
// Address: 0x17fb9e0
//
__int64 sub_17FB9E0()
{
  __int64 result; // rax

  result = sub_22077B0(992);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 80) = result + 64;
    *(_QWORD *)(result + 88) = result + 64;
    *(_QWORD *)(result + 16) = &unk_4FA676C;
    *(_DWORD *)(result + 24) = 3;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)(result + 40) = 0;
    *(_QWORD *)(result + 48) = 0;
    *(_DWORD *)(result + 64) = 0;
    *(_QWORD *)(result + 72) = 0;
    *(_QWORD *)(result + 96) = 0;
    *(_DWORD *)(result + 112) = 0;
    *(_QWORD *)(result + 120) = 0;
    *(_QWORD *)(result + 128) = result + 112;
    *(_QWORD *)(result + 136) = result + 112;
    *(_QWORD *)(result + 144) = 0;
    *(_BYTE *)(result + 152) = 0;
    *(_QWORD *)result = off_49F07A0;
  }
  return result;
}
