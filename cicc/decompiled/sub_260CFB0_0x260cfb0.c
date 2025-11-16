// Function: sub_260CFB0
// Address: 0x260cfb0
//
__int64 __fastcall sub_260CFB0(unsigned __int64 *a1, unsigned __int64 *a2, unsigned __int64 *a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r9
  unsigned __int64 *v8; // rbx
  __int64 v10; // [rsp-10h] [rbp-50h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v6 = (__int64)(0xAAAAAAAAAAAAAAABLL * (a2 - a1) + 1) / 2;
  v7 = v6
     + ((0xAAAAAAAAAAAAAAABLL * (a2 - a1) + 1 + ((0xAAAAAAAAAAAAAAABLL * (a2 - a1) + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
  v11 = v7 * 8;
  v8 = &a1[v7];
  if ( v6 <= a4 )
  {
    sub_25FB2D0(a1, &a1[v7], a3);
    sub_25FB2D0(v8, a2, a3);
  }
  else
  {
    sub_260CFB0(a1, &a1[v7]);
    sub_260CFB0(v8, a2);
  }
  sub_260C720(a1, v8, (__int64)a2, 0xAAAAAAAAAAAAAAABLL * (v11 >> 3), 0xAAAAAAAAAAAAAAABLL * (a2 - v8), a3, a4);
  return v10;
}
