// Function: sub_DA3860
// Address: 0xda3860
//
__int64 *__fastcall sub_DA3860(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rsi
  __int64 *v3; // r12
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // rax
  _QWORD *v8; // r8
  _QWORD *v9; // rbx
  __int64 v10; // r12
  void *v12; // [rsp+18h] [rbp-D8h]
  __int64 *v13; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD v14[2]; // [rsp+30h] [rbp-C0h] BYREF
  int v15; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v16; // [rsp+44h] [rbp-ACh]

  v16 = a2;
  v2 = v14;
  v14[0] = &v15;
  v14[1] = 0x2000000003LL;
  v15 = 15;
  v13 = 0;
  v3 = sub_C65B40((__int64)(a1 + 129), (__int64)v14, (__int64 *)&v13, (__int64)off_49DEA80);
  if ( !v3 )
  {
    v12 = sub_C65D30((__int64)v14, a1 + 133);
    v6 = v5;
    v7 = sub_A777F0(0x50u, a1 + 133);
    v8 = a1;
    v9 = (_QWORD *)v7;
    if ( v7 )
    {
      v10 = a1[193];
      *(_QWORD *)(v7 + 32) = 0;
      *(_QWORD *)(v7 + 40) = v12;
      *(_QWORD *)(v7 + 48) = v6;
      *(_DWORD *)(v7 + 56) = 65551;
      *(_WORD *)(v7 + 60) = 0;
      *(_QWORD *)(v7 + 8) = 2;
      *(_QWORD *)(v7 + 16) = 0;
      *(_QWORD *)(v7 + 24) = a2;
      if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
      {
        sub_BD73F0(v7 + 8);
        v8 = a1;
      }
      v9[8] = v8;
      v9[9] = v10;
      v3 = v9 + 4;
      *v9 = &unk_49DE8E8;
    }
    v8[193] = v9;
    v2 = v3;
    sub_C657C0(a1 + 129, v3, v13, (__int64)off_49DEA80);
  }
  if ( (int *)v14[0] != &v15 )
    _libc_free(v14[0], v2);
  return v3;
}
