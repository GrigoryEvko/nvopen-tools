// Function: sub_14C8310
// Address: 0x14c8310
//
__int64 __fastcall sub_14C8310(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  int v8; // [rsp+8h] [rbp-28h]
  char v9; // [rsp+10h] [rbp-20h]

  sub_14C8230((__int64)&v7, a2, a4);
  if ( v9 )
  {
    v6 = v7;
    *(_QWORD *)(a1 + 16) = a3;
    *(_BYTE *)(a1 + 24) = 1;
    *(_QWORD *)a1 = v6;
    *(_DWORD *)(a1 + 8) = v8;
  }
  else
  {
    *(_BYTE *)(a1 + 24) = 0;
  }
  return a1;
}
