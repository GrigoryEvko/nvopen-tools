// Function: sub_1F6F230
// Address: 0x1f6f230
//
__int64 __fastcall sub_1F6F230(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  unsigned __int8 *v9; // rax
  unsigned int v10; // r14d
  __int64 v11; // rsi
  __int64 *v12; // r10
  __int64 v13; // r13
  __int128 v15; // [rsp-10h] [rbp-60h]
  __int64 v16; // [rsp+0h] [rbp-50h]
  __int64 *v17; // [rsp+0h] [rbp-50h]
  const void **v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  int v20; // [rsp+18h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 32);
  v8 = v6[1];
  v16 = *v6;
  v7 = v16;
  v9 = *(unsigned __int8 **)(a2 + 40);
  v10 = *v9;
  v18 = (const void **)*((_QWORD *)v9 + 1);
  if ( sub_1D23600(*a1, v16) )
  {
    v11 = *(_QWORD *)(a2 + 72);
    v12 = (__int64 *)*a1;
    v19 = v11;
    if ( v11 )
    {
      v17 = v12;
      sub_1623A60((__int64)&v19, v11, 2);
      v12 = v17;
    }
    *((_QWORD *)&v15 + 1) = v8;
    *(_QWORD *)&v15 = v7;
    v20 = *(_DWORD *)(a2 + 64);
    v13 = sub_1D309E0(v12, 131, (__int64)&v19, v10, v18, 0, a3, a4, a5, v15);
    if ( v19 )
      sub_161E7C0((__int64)&v19, v19);
  }
  else
  {
    v13 = 0;
    if ( *(_WORD *)(v16 + 24) == 131 )
      return **(_QWORD **)(v16 + 32);
  }
  return v13;
}
