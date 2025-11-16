// Function: sub_2EF0B60
// Address: 0x2ef0b60
//
unsigned __int64 __fastcall sub_2EF0B60(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8)
{
  bool v8; // r15
  __int64 *v11; // rax
  __int64 *v12; // rax
  unsigned __int64 result; // rax
  __int128 v16; // [rsp+8h] [rbp-48h]
  unsigned __int64 v17; // [rsp+18h] [rbp-38h]

  v8 = 0;
  v17 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  *((_QWORD *)&v16 + 1) = a7;
  *(_QWORD *)&v16 = a8;
  v11 = (__int64 *)sub_2E09D00((__int64 *)a5, a4 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v11 != (__int64 *)(*(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8)) )
  {
    v8 = 0;
    if ( (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v11 >> 1) & 3) <= *(_DWORD *)(v17 + 24) )
      v8 = v17 == (v11[1] & 0xFFFFFFFFFFFFFFF8LL);
  }
  v12 = (__int64 *)sub_2E09D00((__int64 *)a5, a4);
  if ( (v12 == (__int64 *)(*(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8))
     || (*(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v12 >> 1) & 3) > (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                           | (unsigned int)(a4 >> 1) & 3))
    && v16 == 0 )
  {
    sub_2EF0A60(a1, "No live segment at use", a2, a3, 0);
    sub_2EEF4F0(*(_QWORD *)(a1 + 16), a5);
    sub_2EEFA20(a1, a6);
    sub_2EEF640(*(_QWORD *)(a1 + 16), a4);
  }
  result = (*(_BYTE *)(a2 + 3) & 0x40) != 0;
  if ( ((unsigned __int8)result & ((*(_BYTE *)(a2 + 3) >> 4) ^ 1)) != 0 && !v8 )
  {
    sub_2EF0A60(a1, "Live range continues after kill flag", a2, a3, 0);
    sub_2EEF4F0(*(_QWORD *)(a1 + 16), a5);
    sub_2EEFA20(a1, a6);
    if ( v16 != 0 )
      sub_2EEF800(*(_QWORD *)(a1 + 16), a7, a8);
    return (unsigned __int64)sub_2EEF640(*(_QWORD *)(a1 + 16), a4);
  }
  return result;
}
