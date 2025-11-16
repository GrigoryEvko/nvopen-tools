// Function: sub_2F25260
// Address: 0x2f25260
//
void __fastcall sub_2F25260(__int64 a1, __int64 a2, __m128i a3)
{
  char v5; // r14
  __int64 m; // r12
  __int64 v7; // rdi
  __int64 *v8; // rsi
  __int64 j; // r12
  __int64 v10; // rdi
  __int64 k; // r15
  __int64 v12; // rdi
  __int64 i; // r15
  __int64 v14; // rdi
  __int64 v15[2]; // [rsp+10h] [rbp-120h] BYREF
  _QWORD *v16; // [rsp+20h] [rbp-110h] BYREF
  __int64 v17; // [rsp+28h] [rbp-108h]
  _QWORD v18[2]; // [rsp+30h] [rbp-100h] BYREF
  void *v19; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v20; // [rsp+48h] [rbp-E8h]
  __int64 v21; // [rsp+50h] [rbp-E0h]
  __int64 v22; // [rsp+58h] [rbp-D8h]
  __int64 v23; // [rsp+60h] [rbp-D0h]
  __int64 v24; // [rsp+68h] [rbp-C8h]
  _QWORD **v25; // [rsp+70h] [rbp-C0h]
  _QWORD v26[22]; // [rsp+80h] [rbp-B0h] BYREF

  v5 = *(_BYTE *)(a2 + 872);
  if ( unk_4F81788 )
  {
    if ( !v5 )
    {
      for ( i = *(_QWORD *)(a2 + 32); a2 + 24 != i; i = *(_QWORD *)(i + 8) )
      {
        v14 = i - 56;
        if ( !i )
          v14 = 0;
        sub_B2B950(v14);
      }
      *(_BYTE *)(a2 + 872) = 1;
LABEL_3:
      sub_CB1A80((__int64)v26, a1, 0, 70);
      sub_CB2850((__int64)v26);
      if ( !(unsigned __int8)sub_CB2870((__int64)v26, 0) )
      {
        sub_CB1B70((__int64)v26);
        sub_CB0A00(v26, 0);
        if ( !*(_BYTE *)(a2 + 872) )
          return;
LABEL_20:
        for ( j = *(_QWORD *)(a2 + 32); a2 + 24 != j; j = *(_QWORD *)(j + 8) )
        {
          v10 = j - 56;
          if ( !j )
            v10 = 0;
          sub_B2B9A0(v10);
        }
        *(_BYTE *)(a2 + 872) = 0;
        return;
      }
      goto LABEL_15;
    }
  }
  else
  {
    if ( !v5 )
      goto LABEL_3;
    for ( k = *(_QWORD *)(a2 + 32); a2 + 24 != k; k = *(_QWORD *)(k + 8) )
    {
      v12 = k - 56;
      if ( !k )
        v12 = 0;
      sub_B2B9A0(v12);
    }
    *(_BYTE *)(a2 + 872) = 0;
  }
  sub_CB1A80((__int64)v26, a1, 0, 70);
  sub_CB2850((__int64)v26);
  if ( !(unsigned __int8)sub_CB2870((__int64)v26, 0) )
  {
    sub_CB1B70((__int64)v26);
    sub_CB0A00(v26, 0);
    goto LABEL_9;
  }
LABEL_15:
  if ( !(unsigned __int8)sub_CB0090(v26) )
  {
    v19 = 0;
    v20 = 0;
    sub_CB23B0((__int64)v26, (__int64 *)&v19);
    sub_CB0A70((__int64)v26);
    BUG();
  }
  v24 = 0x100000000LL;
  v25 = &v16;
  v19 = &unk_49DD210;
  v16 = v18;
  v17 = 0;
  LOBYTE(v18[0]) = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  sub_CB5980((__int64)&v19, 0, 0, 0);
  sub_CB0A70((__int64)v26);
  sub_A69980((__int64 (__fastcall **)())a2, (__int64)&v19, 0, 0, 0, a3);
  v8 = v15;
  v15[0] = (__int64)v16;
  v15[1] = v17;
  sub_CB23B0((__int64)v26, v15);
  v19 = &unk_49DD210;
  sub_CB5840((__int64)&v19);
  if ( v16 != v18 )
  {
    v8 = (__int64 *)(v18[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v16);
  }
  nullsub_173();
  sub_CB1B70((__int64)v26);
  sub_CB0A00(v26, (__int64)v8);
  if ( !v5 )
  {
    if ( !*(_BYTE *)(a2 + 872) )
      return;
    goto LABEL_20;
  }
LABEL_9:
  if ( !*(_BYTE *)(a2 + 872) )
  {
    for ( m = *(_QWORD *)(a2 + 32); a2 + 24 != m; m = *(_QWORD *)(m + 8) )
    {
      v7 = m - 56;
      if ( !m )
        v7 = 0;
      sub_B2B950(v7);
    }
    *(_BYTE *)(a2 + 872) = 1;
  }
}
