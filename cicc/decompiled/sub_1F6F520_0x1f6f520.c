// Function: sub_1F6F520
// Address: 0x1f6f520
//
__int64 __fastcall sub_1F6F520(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v6; // rax
  __int64 v7; // r13
  __int64 v8; // r12
  unsigned __int8 *v9; // rax
  unsigned int v10; // r14d
  __int64 v11; // rsi
  __int64 *v12; // r10
  __int64 v13; // r14
  __int128 v15; // [rsp-10h] [rbp-70h]
  const void **v16; // [rsp+10h] [rbp-50h]
  __int64 *v17; // [rsp+18h] [rbp-48h]
  __int64 v18; // [rsp+20h] [rbp-40h] BYREF
  int v19; // [rsp+28h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 32);
  v7 = v6[1];
  v8 = *v6;
  v9 = *(unsigned __int8 **)(a2 + 40);
  v10 = *v9;
  v16 = (const void **)*((_QWORD *)v9 + 1);
  if ( sub_1D23600(*a1, v8) )
  {
    v11 = *(_QWORD *)(a2 + 72);
    v12 = (__int64 *)*a1;
    v18 = v11;
    if ( v11 )
    {
      v17 = v12;
      sub_1623A60((__int64)&v18, v11, 2);
      v12 = v17;
    }
    *((_QWORD *)&v15 + 1) = v7;
    *(_QWORD *)&v15 = v8;
    v19 = *(_DWORD *)(a2 + 64);
    v13 = sub_1D309E0(v12, 121, (__int64)&v18, v10, v16, 0, a3, a4, a5, v15);
    if ( v18 )
      sub_161E7C0((__int64)&v18, v18);
  }
  else if ( *(_WORD *)(v8 + 24) == 121 || (unsigned __int8)sub_1D1F9F0(*a1, v8, v7, 0) )
  {
    return v8;
  }
  else
  {
    return 0;
  }
  return v13;
}
