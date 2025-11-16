// Function: sub_161BC40
// Address: 0x161bc40
//
__int64 __fastcall sub_161BC40(unsigned __int64 *src, unsigned __int64 *a2, char *a3, char *a4)
{
  __int64 v6; // rax
  unsigned __int64 *v7; // r15
  __int64 v9; // [rsp-10h] [rbp-50h]
  __int64 v10; // [rsp+8h] [rbp-38h]

  v6 = (a2 - src + 1) / 2;
  v10 = 8 * v6;
  v7 = &src[v6];
  if ( v6 <= (__int64)a4 )
  {
    sub_161B2F0(src, &src[v6], a3);
    sub_161B2F0(v7, a2, a3);
  }
  else
  {
    sub_161BC40(src);
    sub_161BC40(v7);
  }
  sub_161B7E0((char *)src, (char *)v7, (__int64)a2, v10 >> 3, a2 - v7, a3, a4);
  return v9;
}
