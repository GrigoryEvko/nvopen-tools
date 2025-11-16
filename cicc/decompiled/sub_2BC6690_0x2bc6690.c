// Function: sub_2BC6690
// Address: 0x2bc6690
//
void *__fastcall sub_2BC6690(
        __int64 *src,
        __int64 *a2,
        char *a3,
        void *a4,
        __int64 (__fastcall *a5)(__int64, __int64, __int64),
        __int64 a6)
{
  __int64 v9; // rax
  char *v10; // r10
  void **v11; // rdx
  __int64 v12; // r11
  __int128 v14; // [rsp-18h] [rbp-68h]
  __int64 v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int64 *srca; // [rsp+18h] [rbp-38h]
  __int64 *srcb; // [rsp+18h] [rbp-38h]

  v9 = (a2 - src + 1) / 2;
  if ( v9 <= (__int64)a4 )
  {
    v15 = 8 * v9;
    srcb = &src[v9];
    sub_2BC6590(src, srcb, a3, a5, a6);
    sub_2BC6590(srcb, a2, a3, a5, a6);
    v12 = v15;
    v11 = (void **)a3;
    v10 = (char *)srcb;
  }
  else
  {
    v16 = 8 * v9;
    srca = &src[v9];
    sub_2BC6690(src);
    sub_2BC6690(srca);
    v10 = (char *)srca;
    v11 = (void **)a3;
    v12 = v16;
  }
  *((_QWORD *)&v14 + 1) = a6;
  *(_QWORD *)&v14 = a5;
  return sub_2BB7660((char *)src, v10, (__int64)a2, v12 >> 3, ((char *)a2 - v10) >> 3, v11, a4, v14);
}
