// Function: sub_16A2F10
// Address: 0x16a2f10
//
_QWORD *__fastcall sub_16A2F10(_QWORD *a1, __int64 a2, _DWORD *a3, unsigned int a4)
{
  __int64 v5; // r15
  __int16 *v6; // rax
  __int16 *v7; // r14
  __int64 *v8; // rsi
  __int16 *v9; // rbx
  __int64 *v10; // rsi
  char *v12; // rax
  __int64 v13; // rsi
  __int64 *v14; // r14
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // r14
  __int64 *v18; // [rsp+8h] [rbp-128h]
  __int64 v19; // [rsp+8h] [rbp-128h]
  __int16 *v20; // [rsp+8h] [rbp-128h]
  int v23; // [rsp+18h] [rbp-118h]
  __int16 *v24; // [rsp+20h] [rbp-110h] BYREF
  __int64 v25[3]; // [rsp+28h] [rbp-108h] BYREF
  __int64 v26; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v27[3]; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v28; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v29[3]; // [rsp+68h] [rbp-C8h] BYREF
  char v30[8]; // [rsp+80h] [rbp-B0h] BYREF
  _QWORD v31[3]; // [rsp+88h] [rbp-A8h] BYREF
  char v32[8]; // [rsp+A0h] [rbp-90h] BYREF
  void *v33[3]; // [rsp+A8h] [rbp-88h] BYREF
  __int64 v34; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v35[3]; // [rsp+C8h] [rbp-68h] BYREF
  __int64 v36; // [rsp+E0h] [rbp-50h] BYREF
  void *v37[9]; // [rsp+E8h] [rbp-48h] BYREF

  v5 = *(_QWORD *)(a2 + 8);
  v6 = (__int16 *)sub_16982C0();
  v7 = *(__int16 **)(v5 + 8);
  v8 = (__int64 *)(v5 + 8);
  v9 = v6;
  if ( v7 == v6 )
  {
    sub_16A2F10(&v34, v8, a3, a4);
    sub_169C7E0(&v36, &v34);
    sub_169C7E0(v31, &v36);
    v12 = (char *)v37[0];
    if ( v37[0] )
    {
      v13 = 32LL * *((_QWORD *)v37[0] - 1);
      v14 = (__int64 *)((char *)v37[0] + v13);
      if ( v37[0] != (char *)v37[0] + v13 )
      {
        do
        {
          v14 -= 4;
          v18 = (__int64 *)v12;
          if ( v9 == (__int16 *)v14[1] )
            sub_169DEB0(v14 + 2);
          else
            sub_1698460((__int64)(v14 + 1));
          v12 = (char *)v18;
        }
        while ( v18 != v14 );
      }
      j_j_j___libc_free_0_0(v12 - 8);
    }
    v15 = v35[0];
    if ( v35[0] )
    {
      v16 = 32LL * *(_QWORD *)(v35[0] - 8);
      v17 = v35[0] + v16;
      if ( v35[0] != v35[0] + v16 )
      {
        do
        {
          v17 -= 32;
          v19 = v15;
          if ( v9 == *(__int16 **)(v17 + 8) )
            sub_169DEB0((__int64 *)(v17 + 16));
          else
            sub_1698460(v17 + 8);
          v15 = v19;
        }
        while ( v19 != v17 );
      }
      j_j_j___libc_free_0_0(v15 - 8);
    }
  }
  else
  {
    sub_169C410(&v34, v8, a3, a4);
    sub_1698450((__int64)&v36, (__int64)&v34);
    sub_169E320(v31, &v36, v7);
    sub_1698460((__int64)&v36);
    sub_1698460((__int64)&v34);
  }
  v10 = (__int64 *)(*(_QWORD *)(a2 + 8) + 40LL);
  if ( (__int16 *)*v10 == v9 )
  {
    sub_169C6E0(v33, (__int64)v10);
    if ( (unsigned int)sub_169C920(a2) != 2 )
      goto LABEL_5;
  }
  else
  {
    sub_16986C0(v33, v10);
    if ( (unsigned int)sub_169C920(a2) != 2 )
      goto LABEL_5;
  }
  v23 = -*a3;
  if ( v9 == v33[0] )
    sub_169C6E0(v35, (__int64)v33);
  else
    sub_16986C0(v35, (__int64 *)v33);
  v20 = (__int16 *)v35[0];
  if ( v9 == (__int16 *)v35[0] )
  {
    sub_169C6E0(&v24, (__int64)v35);
    sub_16A28C0(&v26, (__int64)&v24, v23, a4);
    sub_169C7E0(&v28, &v26);
    sub_169C7E0(v37, &v28);
    sub_169DEB0(v29);
    sub_169DEB0(v27);
    sub_169DEB0(v25);
  }
  else
  {
    sub_16986C0(&v24, v35);
    sub_169C390((__int64)&v26, &v24, v23, a4);
    sub_1698450((__int64)&v28, (__int64)&v26);
    sub_169E320(v37, &v28, v20);
    sub_1698460((__int64)&v28);
    sub_1698460((__int64)&v26);
    sub_1698460((__int64)&v24);
  }
  sub_169ED90(v33, v37);
  sub_127D120(v37);
  sub_127D120(v35);
LABEL_5:
  sub_169C810(a1, (__int64)&unk_42AE990, (__int64)v30, (__int64)v32);
  sub_127D120(v33);
  sub_127D120(v31);
  return a1;
}
