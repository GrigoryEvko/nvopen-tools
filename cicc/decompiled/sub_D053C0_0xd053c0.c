// Function: sub_D053C0
// Address: 0xd053c0
//
__int64 __fastcall sub_D053C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // rax

  v6 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v7 = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  v8 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v9 = sub_B2BEC0(a3);
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)a1 = v9;
  *(_QWORD *)(a1 + 16) = v6 + 8;
  *(_QWORD *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 32) = v8 + 8;
  *(_QWORD *)(a1 + 48) = a1 + 72;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 16;
  *(_DWORD *)(a1 + 64) = 0;
  *(_BYTE *)(a1 + 68) = 1;
  return a1;
}
