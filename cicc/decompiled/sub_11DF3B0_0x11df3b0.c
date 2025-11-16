// Function: sub_11DF3B0
// Address: 0x11df3b0
//
__int64 __fastcall sub_11DF3B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // r13
  unsigned __int64 v5; // rax
  __int64 v6; // r15
  _QWORD **v7; // rbx
  unsigned int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int8 *v11; // rax
  __int64 *v13; // [rsp+0h] [rbp-50h]
  unsigned int v15[14]; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( v4 != v3 )
  {
    *(_QWORD *)v15 = 0x100000000LL;
    sub_11DA4B0(a2, (int *)v15, 2);
    v5 = sub_98B430(v4, 8u);
    v6 = v5;
    if ( v5 )
    {
      v15[0] = 1;
      sub_11DA2E0(a2, v15, 1, v5);
      v13 = *(__int64 **)(a1 + 24);
      v7 = (_QWORD **)sub_B43CA0(a2);
      v8 = sub_97FA80(*v13, (__int64)v7);
      v9 = sub_BCCE00(*v7, v8);
      v10 = sub_ACD640(v9, v6, 0);
      v11 = (unsigned __int8 *)sub_B343C0(a3, 0xEEu, v3, 0x100u, v4, 0x100u, v10, 0, 0, 0, 0, 0);
      sub_11DAF00(v11, a2);
    }
    else
    {
      return 0;
    }
  }
  return v3;
}
