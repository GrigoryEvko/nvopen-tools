// Function: sub_1002AD0
// Address: 0x1002ad0
//
__int64 __fastcall sub_1002AD0(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned __int8 *v5; // rax
  __int64 v6; // r14
  __int64 v7; // r15
  unsigned __int8 *v8; // r12
  unsigned int v9; // eax
  _BYTE *v10; // rcx
  __int64 v11; // rsi
  char *v12; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-48h]
  char *v14; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-38h]

  if ( (*(_BYTE *)(a1 + 2) & 1) != 0 )
    return 0;
  if ( *(_BYTE *)a2 <= 0x15u )
    return sub_9718F0(a2, *(_QWORD *)(a1 + 8), (_BYTE *)*a3);
  v5 = sub_98ACB0((unsigned __int8 *)a2, 6u);
  v6 = (__int64)v5;
  if ( *v5 != 3
    || (v5[80] & 1) == 0
    || sub_B2FC80((__int64)v5)
    || (unsigned __int8)sub_B2F6B0(v6)
    || (*(_BYTE *)(v6 + 80) & 2) != 0 )
  {
    return 0;
  }
  v7 = sub_96E500(*(unsigned __int8 **)(v6 - 32), *(_QWORD *)(a1 + 8), *a3);
  if ( !v7 )
  {
    v13 = sub_AE43F0(*a3, *(_QWORD *)(a2 + 8));
    if ( v13 > 0x40 )
      sub_C43690((__int64)&v12, 0, 0);
    else
      v12 = 0;
    v8 = sub_BD45C0((unsigned __int8 *)a2, *a3, (__int64)&v12, 1, 1, 0, 0, 0);
    if ( (unsigned __int8 *)v6 == v8 )
    {
      v9 = sub_AE43F0(*a3, *((_QWORD *)v8 + 1));
      sub_C44B10((__int64)&v14, &v12, v9);
      if ( v13 > 0x40 && v12 )
        j_j___libc_free_0_0(v12);
      v10 = (_BYTE *)*a3;
      v11 = *(_QWORD *)(a1 + 8);
      v13 = 0;
      v12 = v14;
      v7 = sub_971820((__int64)v8, v11, (__int64)&v14, v10);
      if ( v15 > 0x40 && v14 )
        j_j___libc_free_0_0(v14);
    }
    if ( v13 > 0x40 )
    {
      if ( v12 )
        j_j___libc_free_0_0(v12);
    }
  }
  return v7;
}
