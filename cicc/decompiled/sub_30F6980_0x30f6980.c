// Function: sub_30F6980
// Address: 0x30f6980
//
__int64 __fastcall sub_30F6980(__int64 *a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r9
  __int64 *v8; // rbx
  __int64 v10; // [rsp-10h] [rbp-50h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  v6 = (__int64)(0xAAAAAAAAAAAAAAABLL * (a2 - a1) + 1) / 2;
  v7 = v6
     + ((0xAAAAAAAAAAAAAAABLL * (a2 - a1) + 1 + ((0xAAAAAAAAAAAAAAABLL * (a2 - a1) + 1) >> 63)) & 0xFFFFFFFFFFFFFFFELL);
  v11 = v7 * 8;
  v8 = &a1[v7];
  if ( v6 <= a4 )
  {
    sub_30F3BC0(a1, &a1[v7], a3);
    sub_30F3BC0(v8, a2, a3);
  }
  else
  {
    sub_30F6980(a1, &a1[v7]);
    sub_30F6980(v8, a2);
  }
  sub_30F64D0(
    (__int64)a1,
    (__int64)v8,
    (__int64)a2,
    0xAAAAAAAAAAAAAAABLL * (v11 >> 3),
    0xAAAAAAAAAAAAAAABLL * (a2 - v8),
    (__int64)a3,
    a4);
  return v10;
}
