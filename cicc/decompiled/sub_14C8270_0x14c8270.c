// Function: sub_14C8270
// Address: 0x14c8270
//
__int64 __fastcall sub_14C8270(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-50h] BYREF
  int v12; // [rsp+8h] [rbp-48h]
  char v13; // [rsp+10h] [rbp-40h]
  __int64 v14; // [rsp+20h] [rbp-30h] BYREF
  int v15; // [rsp+28h] [rbp-28h]
  char v16; // [rsp+30h] [rbp-20h]

  sub_14C8230((__int64)&v11, a7, a2);
  if ( v13 && (sub_14C8230((__int64)&v14, a8, a2), v16) )
  {
    v10 = v11;
    *(_BYTE *)(a1 + 40) = 1;
    *(_QWORD *)a1 = v10;
    *(_DWORD *)(a1 + 8) = v12;
    *(_QWORD *)(a1 + 16) = v14;
    *(_DWORD *)(a1 + 24) = v15;
    *(_QWORD *)(a1 + 32) = a9;
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 40) = 0;
    return a1;
  }
}
