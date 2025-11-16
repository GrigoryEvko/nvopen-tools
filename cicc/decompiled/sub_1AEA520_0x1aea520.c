// Function: sub_1AEA520
// Address: 0x1aea520
//
__int64 __fastcall sub_1AEA520(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, char a5, int a6, char a7)
{
  __int64 v10; // rdx
  unsigned int v11; // r12d
  unsigned __int64 v12; // r13
  __int64 *v14; // rax
  char v15; // di
  __int64 *v16; // r13
  __int64 v17; // r14
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 *v23; // [rsp+0h] [rbp-70h]
  __int64 v24; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h] BYREF
  __int64 v29[7]; // [rsp+38h] [rbp-38h] BYREF

  sub_1AEA030(&v28, a1);
  v10 = v28;
  if ( (v28 & 4) != 0 )
  {
    v14 = *(__int64 **)(v28 & 0xFFFFFFFFFFFFFFF8LL);
    v23 = &v14[*(unsigned int *)((v28 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
    if ( v14 == v23 )
      goto LABEL_3;
  }
  else
  {
    if ( (v28 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_3;
    v14 = &v28;
    v23 = v29;
  }
  v15 = a5;
  v16 = v14;
  v24 = a6;
  do
  {
    v17 = *v16;
    v18 = *(_QWORD *)(*v16 + 48);
    v29[0] = v18;
    if ( v18 )
      sub_1623A60((__int64)v29, v18, 2);
    v19 = *(_DWORD *)(v17 + 20) & 0xFFFFFFF;
    v20 = *(_QWORD *)(*(_QWORD *)(v17 + 24 * (1 - v19)) + 24LL);
    v27 = sub_15C48E0(*(_QWORD **)(*(_QWORD *)(v17 + 24 * (2 - v19)) + 24LL), v15, v24, a7, 0);
    v21 = sub_15C70A0((__int64)v29);
    sub_15A7500(a4, a2, v20, v27, v21, a3);
    if ( v17 == a3 )
    {
      v22 = *(_QWORD *)(v17 + 32);
      if ( v22 == *(_QWORD *)(v17 + 40) + 40LL || !v22 )
        a3 = 0;
      else
        a3 = v22 - 24;
    }
    sub_15F20C0((_QWORD *)v17);
    if ( v29[0] )
      sub_161E7C0((__int64)v29, v29[0]);
    ++v16;
  }
  while ( v23 != v16 );
  v10 = v28;
LABEL_3:
  v11 = 0;
  v12 = v10 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v10 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v11 = 1;
    if ( (v10 & 4) != 0 )
    {
      LOBYTE(v11) = *(_DWORD *)(v12 + 8) != 0;
      if ( *(_QWORD *)v12 != v12 + 16 )
        _libc_free(*(_QWORD *)v12);
      j_j___libc_free_0(v12, 48);
    }
  }
  return v11;
}
