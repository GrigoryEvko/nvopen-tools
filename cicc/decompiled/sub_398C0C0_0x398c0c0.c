// Function: sub_398C0C0
// Address: 0x398c0c0
//
__int64 __fastcall sub_398C0C0(__int64 a1, unsigned int a2, unsigned int a3, _BYTE *a4, unsigned int a5)
{
  unsigned __int16 v8; // ax
  __int64 v10; // [rsp-10h] [rbp-50h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v11 = *(_QWORD *)(a1 + 4208);
  v8 = sub_398C0A0(a1);
  sub_3987590(
    *(_QWORD ***)(a1 + 8),
    a2,
    a3,
    a4,
    a5,
    *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL) + 8LL) + 1164LL),
    v8,
    v11);
  return v10;
}
