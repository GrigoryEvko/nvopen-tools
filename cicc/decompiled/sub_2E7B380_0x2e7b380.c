// Function: sub_2E7B380
// Address: 0x2e7b380
//
_QWORD *__fastcall sub_2E7B380(_QWORD *a1, __int64 a2, unsigned __int8 **a3, unsigned __int8 a4)
{
  unsigned __int8 *v6; // rsi
  _QWORD *v8; // r13
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  __int64 v12[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = *a3;
  v12[0] = (__int64)v6;
  if ( v6 )
  {
    sub_B976B0((__int64)a3, v6, (__int64)v12);
    *a3 = 0;
  }
  v8 = (_QWORD *)a1[28];
  if ( v8 )
  {
    a1[28] = *v8;
LABEL_5:
    sub_2E91000(v8, a1, a2, v12, a4);
    goto LABEL_6;
  }
  v10 = a1[16];
  a1[26] += 72LL;
  v11 = (v10 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[17] < v11 + 72 || !v10 )
  {
    v11 = sub_9D1E70((__int64)(a1 + 16), 72, 72, 3);
LABEL_14:
    v8 = (_QWORD *)v11;
    goto LABEL_5;
  }
  a1[16] = v11 + 72;
  if ( v11 )
    goto LABEL_14;
LABEL_6:
  if ( v12[0] )
    sub_B91220((__int64)v12, v12[0]);
  return v8;
}
