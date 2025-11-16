// Function: sub_D174E0
// Address: 0xd174e0
//
__int64 __fastcall sub_D174E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rax

  v6 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v7 = sub_BC1CD0(a4, &unk_4F81450, a3);
  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = v6 + 8;
  *(_QWORD *)(a1 + 16) = v7 + 8;
  *(_QWORD *)(a1 + 40) = a1 + 64;
  *(_QWORD *)(a1 + 360) = a1 + 384;
  *(_BYTE *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = 32;
  *(_DWORD *)(a1 + 56) = 0;
  *(_BYTE *)(a1 + 60) = 1;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_DWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 368) = 16;
  *(_DWORD *)(a1 + 376) = 0;
  *(_BYTE *)(a1 + 380) = 1;
  return a1;
}
