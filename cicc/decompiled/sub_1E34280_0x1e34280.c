// Function: sub_1E34280
// Address: 0x1e34280
//
__int64 __fastcall sub_1E34280(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = sub_1E0A0C0(a2);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 0;
  LODWORD(v2) = *(_DWORD *)(v2 + 4);
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 20) = v2;
  return a1;
}
