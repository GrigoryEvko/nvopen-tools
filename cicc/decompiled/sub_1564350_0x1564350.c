// Function: sub_1564350
// Address: 0x1564350
//
__int64 __fastcall sub_1564350(__int64 a1, unsigned int a2, _QWORD *a3)
{
  unsigned int v4; // r12d
  __int64 v5; // rdx
  _QWORD v7[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v8[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v9; // [rsp+20h] [rbp-30h]

  v4 = sub_1642F90(
         *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL)
                   + 8LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 24) + 12LL) - 1)),
         32);
  if ( (_BYTE)v4 )
  {
    v7[0] = sub_1649960(a1);
    v9 = 773;
    v8[0] = v7;
    v7[1] = v5;
    v8[1] = ".old";
    sub_164B780(a1, v8);
    *a3 = sub_15E26F0(*(_QWORD *)(a1 + 40), a2, 0, 0);
  }
  return v4;
}
