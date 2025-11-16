// Function: sub_2BB9A80
// Address: 0x2bb9a80
//
__int64 __fastcall sub_2BB9A80(unsigned __int64 *a1, unsigned int *a2, unsigned __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // r9
  unsigned __int64 *v9; // r10
  __int64 v10; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // [rsp-10h] [rbp-50h]
  __int64 v15; // [rsp+0h] [rbp-40h]
  unsigned __int64 *v16; // [rsp+8h] [rbp-38h]

  v8 = (((((char *)a2 - (char *)a1) >> 6) + 1) / 2) << 6;
  v15 = v8;
  v16 = (unsigned __int64 *)((char *)a1 + v8);
  if ( ((((char *)a2 - (char *)a1) >> 6) + 1) / 2 <= a4 )
  {
    sub_2BB8A30(a1, (unsigned int *)((char *)a1 + v8), a3, a5, a5, v8);
    sub_2BB8A30(v16, a2, a3, a5, v12, v13);
    v10 = v15;
    v9 = v16;
  }
  else
  {
    sub_2BB9A80(a1, (char *)a1 + v8, a3);
    sub_2BB9A80(v16, a2, a3);
    v9 = v16;
    v10 = v15;
  }
  sub_2BB93D0((__int64)a1, v9, (__int64)a2, v10 >> 6, ((char *)a2 - (char *)v9) >> 6, (__int64)a3, a4, a5);
  return v14;
}
