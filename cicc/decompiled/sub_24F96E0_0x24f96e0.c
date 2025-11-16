// Function: sub_24F96E0
// Address: 0x24f96e0
//
bool __fastcall sub_24F96E0(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  __int64 v5; // r14
  _QWORD *v6; // rbx
  _QWORD *v7; // rax
  _QWORD v9[7]; // [rsp+8h] [rbp-38h] BYREF

  v9[0] = a2;
  v4 = sub_24F9690((__int64)a1, v9);
  v5 = *a1;
  v9[0] = a3;
  v6 = v4;
  v7 = sub_24F9690((__int64)a1, v9);
  return (*(_QWORD *)(*(_QWORD *)(a1[34]
                                + 8
                                * ((((__int64)v7 - v5) >> 3)
                                 + 2
                                 * ((((__int64)v7 - v5) >> 3) + (((unsigned __int64)v7 - v5) & 0xFFFFFFFFFFFFFFF8LL)))
                                + 72)
                    + 8LL * ((unsigned int)(((__int64)v6 - v5) >> 3) >> 6))
        & (1LL << (((__int64)v6 - v5) >> 3))) != 0;
}
