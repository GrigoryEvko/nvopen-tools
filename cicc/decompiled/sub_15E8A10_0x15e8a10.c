// Function: sub_15E8A10
// Address: 0x15e8a10
//
__int64 __fastcall sub_15E8A10(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        unsigned int a6,
        char *a7,
        __int64 a8,
        char *a9,
        __int64 a10,
        char *a11,
        __int64 a12,
        char *a13,
        __int64 a14)
{
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  _BYTE *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v28; // [rsp+38h] [rbp-48h] BYREF
  _QWORD v29[7]; // [rsp+48h] [rbp-38h] BYREF

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  v17 = *(_QWORD *)(a2 + 24);
  v28 = a5;
  v18 = sub_1643360(v17);
  v29[0] = sub_159C470(v18, a3, 0);
  sub_15E88C0(a1, v29);
  v19 = sub_1643350(*(_QWORD *)(a2 + 24));
  v29[0] = sub_159C470(v19, a4, 0);
  sub_15E88C0(a1, v29);
  v20 = *(_BYTE **)(a1 + 8);
  if ( v20 == *(_BYTE **)(a1 + 16) )
  {
    sub_1287830(a1, v20, &v28);
  }
  else
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = v28;
      v20 = *(_BYTE **)(a1 + 8);
    }
    *(_QWORD *)(a1 + 8) = v20 + 8;
  }
  v21 = sub_1643350(*(_QWORD *)(a2 + 24));
  v29[0] = sub_159C470(v21, (unsigned int)a8, 0);
  sub_15E88C0(a1, v29);
  v22 = sub_1643350(*(_QWORD *)(a2 + 24));
  v29[0] = sub_159C470(v22, a6, 0);
  sub_15E88C0(a1, v29);
  sub_15E6650(a1, *(char **)(a1 + 8), a7, &a7[24 * a8]);
  v23 = sub_1643350(*(_QWORD *)(a2 + 24));
  v29[0] = sub_159C470(v23, (unsigned int)a10, 0);
  sub_15E88C0(a1, v29);
  sub_15E6650(a1, *(char **)(a1 + 8), a9, &a9[24 * a10]);
  v24 = sub_1643350(*(_QWORD *)(a2 + 24));
  v29[0] = sub_159C470(v24, (unsigned int)a12, 0);
  sub_15E88C0(a1, v29);
  sub_15E6650(a1, *(char **)(a1 + 8), a11, &a11[24 * a12]);
  sub_15E69B0(a1, *(char **)(a1 + 8), a13, &a13[8 * a14]);
  return a1;
}
