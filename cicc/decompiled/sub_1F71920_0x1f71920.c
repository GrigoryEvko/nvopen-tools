// Function: sub_1F71920
// Address: 0x1f71920
//
__int64 __fastcall sub_1F71920(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v6; // rax
  __int64 v7; // r12
  _QWORD *v8; // r13
  unsigned __int8 *v9; // rax
  unsigned int v10; // r15d
  unsigned int v11; // ecx
  __int64 v12; // rsi
  __int64 *v13; // r10
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 *v20; // r10
  __int128 v21; // [rsp-10h] [rbp-60h]
  __int128 v22; // [rsp-10h] [rbp-60h]
  const void **v23; // [rsp+0h] [rbp-50h]
  unsigned __int8 v24; // [rsp+8h] [rbp-48h]
  __int64 *v25; // [rsp+8h] [rbp-48h]
  __int64 *v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h] BYREF
  int v28; // [rsp+18h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 32);
  v7 = *v6;
  v8 = (_QWORD *)v6[1];
  v9 = *(unsigned __int8 **)(a2 + 40);
  v24 = *v9;
  v10 = *v9;
  v23 = (const void **)*((_QWORD *)v9 + 1);
  if ( sub_1D23600(*(_QWORD *)a1, v7) )
  {
    v12 = *(_QWORD *)(a2 + 72);
    v13 = *(__int64 **)a1;
    v27 = v12;
    if ( v12 )
    {
      v25 = v13;
      sub_1623A60((__int64)&v27, v12, 2);
      v13 = v25;
    }
    *((_QWORD *)&v21 + 1) = v8;
    *(_QWORD *)&v21 = v7;
    v28 = *(_DWORD *)(a2 + 64);
    v14 = sub_1D309E0(v13, 129, (__int64)&v27, v10, v23, 0, a3, a4, a5, v21);
  }
  else
  {
    if ( *(_BYTE *)(a1 + 24) )
    {
      v17 = *(_QWORD *)(a1 + 8);
      v18 = 1;
      if ( v24 != 1 )
      {
        if ( !v24 )
          return 0;
        v18 = v24;
        if ( !*(_QWORD *)(v17 + 8LL * v24 + 120) )
          return 0;
      }
      if ( *(_BYTE *)(v17 + 259 * v18 + 2555) )
        return 0;
    }
    if ( !(unsigned __int8)sub_1D181C0(*(_QWORD *)a1, v7, v8, v11) )
      return 0;
    v19 = *(_QWORD *)(a2 + 72);
    v20 = *(__int64 **)a1;
    v27 = v19;
    if ( v19 )
    {
      v26 = v20;
      sub_1623A60((__int64)&v27, v19, 2);
      v20 = v26;
    }
    *((_QWORD *)&v22 + 1) = v8;
    *(_QWORD *)&v22 = v7;
    v28 = *(_DWORD *)(a2 + 64);
    v14 = sub_1D309E0(v20, 133, (__int64)&v27, v10, v23, 0, a3, a4, a5, v22);
  }
  v15 = v14;
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v15;
}
