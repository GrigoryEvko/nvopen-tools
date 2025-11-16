// Function: sub_2598170
// Address: 0x2598170
//
__int64 __fastcall sub_2598170(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // ebx
  char v5; // [rsp+16h] [rbp-42h] BYREF
  char v6; // [rsp+17h] [rbp-41h] BYREF
  _QWORD v7[8]; // [rsp+18h] [rbp-40h] BYREF

  v2 = sub_25294B0(a2, *(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), a1, 2, 0, 1);
  if ( v2 && (*(_BYTE *)(v2 + 97) & 3) == 3 )
  {
    if ( (*(_BYTE *)(v2 + 96) & 3) == 3 )
      *(_DWORD *)(a1 + 96) = *(_DWORD *)(a1 + 100);
    else
      sub_250ED80(a2, v2, a1, 1);
    return 1;
  }
  else
  {
    v5 = 0;
    v3 = *(_DWORD *)(a1 + 100);
    v7[0] = a2;
    v7[1] = &v5;
    v7[2] = a1;
    v6 = 0;
    if ( (unsigned __int8)sub_25264B0(
                            a2,
                            (unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64))sub_2598870,
                            (__int64)v7,
                            a1,
                            &v6) )
      return (unsigned __int8)(v5 | (*(_DWORD *)(a1 + 100) != v3)) ^ 1u;
    else
      return sub_2562110(a1);
  }
}
