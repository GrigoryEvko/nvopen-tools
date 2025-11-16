// Function: sub_15D2410
// Address: 0x15d2410
//
void __fastcall sub_15D2410(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  _BYTE *v9; // rax
  __int64 v10; // [rsp+10h] [rbp-50h]
  __int64 v11; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+20h] [rbp-40h] BYREF
  char *v13[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_15D1D60((__int64)(a1 + 3), (__int64 *)(*a1 + 8LL))[4] = *a3;
  v3 = *a1;
  v10 = (__int64)(a1[1] - *a1) >> 3;
  if ( v10 != 1 )
  {
    v4 = 1;
    while ( 1 )
    {
      v12 = *(_QWORD *)(v3 + 8 * v4);
      v5 = sub_15CC510(a2, v12);
      v6 = sub_15D1D60((__int64)(a1 + 3), &v12);
      v7 = sub_15CC510(a2, v6[4]);
      v8 = *(_QWORD *)(v5 + 8);
      v11 = v7;
      if ( v7 != v8 )
      {
        v13[0] = (char *)v5;
        v9 = sub_15CBEB0(*(_QWORD **)(v8 + 24), *(_QWORD *)(v8 + 32), (__int64 *)v13);
        sub_15CDF70(*(_QWORD *)(v5 + 8) + 24LL, v9);
        *(_QWORD *)(v5 + 8) = v11;
        v13[0] = (char *)v5;
        sub_15CE4A0(v11 + 24, v13);
        sub_15CC3F0(v5);
      }
      if ( ++v4 == v10 )
        break;
      v3 = *a1;
    }
  }
}
