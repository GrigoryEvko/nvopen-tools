// Function: sub_2ED40C0
// Address: 0x2ed40c0
//
void *__fastcall sub_2ED40C0(__int64 *src, __int64 *a2, __int64 *a3, __int64 a4, __int64 *a5, _QWORD *a6)
{
  __int64 v9; // rax
  __int64 *v10; // r10
  __int64 *v11; // rdx
  __int64 v12; // r11
  __int128 v14; // [rsp-18h] [rbp-68h]
  __int64 v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 *srca; // [rsp+18h] [rbp-38h]
  __int64 *srcb; // [rsp+18h] [rbp-38h]

  v9 = (a2 - src + 1) / 2;
  if ( v9 <= a4 )
  {
    v15 = 8 * v9;
    srcb = &src[v9];
    sub_2ED3560(src, srcb, a3, a5, a6);
    sub_2ED3560(srcb, a2, a3, a5, a6);
    v12 = v15;
    v11 = a3;
    v10 = srcb;
  }
  else
  {
    v16 = 8 * v9;
    srca = &src[v9];
    sub_2ED40C0(src);
    sub_2ED40C0(srca);
    v10 = srca;
    v11 = a3;
    v12 = v16;
  }
  *((_QWORD *)&v14 + 1) = a6;
  *(_QWORD *)&v14 = a5;
  return sub_2ED3A60(src, v10, (char *)a2, v12 >> 3, a2 - v10, v11, a4, v14);
}
