// Function: sub_36FFCE0
// Address: 0x36ffce0
//
void __fastcall sub_36FFCE0(_BYTE *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  _BYTE *v5; // rax
  __int64 v6[2]; // [rsp+0h] [rbp-F0h] BYREF
  unsigned __int64 v7[2]; // [rsp+10h] [rbp-E0h] BYREF
  int v8; // [rsp+20h] [rbp-D0h] BYREF
  char v9; // [rsp+24h] [rbp-CCh]
  __int64 v10; // [rsp+A0h] [rbp-50h]
  __int64 v11; // [rsp+A8h] [rbp-48h]
  __int64 v12; // [rsp+B0h] [rbp-40h]
  __int64 v13; // [rsp+B8h] [rbp-38h]

  v3 = *(_QWORD *)a1;
  v7[0] = (unsigned __int64)&v8;
  v10 = 0;
  v12 = v3;
  v7[1] = 0x1000000001LL;
  v11 = 0;
  v13 = 0;
  v8 = 0;
  v9 = 0;
  sub_C6ACB0((__int64)v7);
  v6[0] = (__int64)a1;
  v6[1] = (__int64)v7;
  sub_C6B410((__int64)v7, (unsigned __int8 *)"features", 8u);
  sub_C6AB50((__int64)v7);
  sub_36FFC90(v6);
  sub_C6AC30((__int64)v7);
  sub_C6AE10((__int64)v7);
  if ( a1[112] )
  {
    sub_C6B410((__int64)v7, (unsigned __int8 *)"score", 5u);
    sub_310D630((__int64)(a1 + 32), (__int64)v7);
    sub_C6AE10((__int64)v7);
    if ( !*(_BYTE *)(a2 + 80) )
      goto LABEL_3;
  }
  else if ( !*(_BYTE *)(a2 + 80) )
  {
    goto LABEL_3;
  }
  sub_C6B410((__int64)v7, "advice", 6u);
  sub_310D630(a2, (__int64)v7);
  sub_C6AE10((__int64)v7);
LABEL_3:
  sub_C6AD90((__int64)v7);
  v4 = *(_QWORD *)a1;
  v5 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
  if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v5 )
  {
    sub_CB6200(v4, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v5 = 10;
    ++*(_QWORD *)(v4 + 32);
  }
  if ( (int *)v7[0] != &v8 )
    _libc_free(v7[0]);
}
