// Function: sub_22BACB0
// Address: 0x22bacb0
//
__int64 __fastcall sub_22BACB0(char *a1, char *a2, unsigned int *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rax
  char *v8; // rbx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // [rsp-10h] [rbp-50h]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v7 = (__int64)(0x8E38E38E38E38E39LL * ((a2 - a1) >> 3) + 1) / 2;
  v14 = 72 * v7;
  v8 = &a1[72 * v7];
  if ( v7 <= a4 )
  {
    sub_22ADDC0(a1, &a1[72 * v7], a3, a4, a5, 72 * v7);
    sub_22ADDC0(v8, a2, a3, v10, v11, v12);
  }
  else
  {
    sub_22BACB0(a1, &a1[72 * v7], a3);
    sub_22BACB0(v8, a2, a3);
  }
  sub_22BA6E0(
    (unsigned int *)a1,
    (unsigned int *)v8,
    (__int64)a2,
    0x8E38E38E38E38E39LL * (v14 >> 3),
    0x8E38E38E38E38E39LL * ((a2 - v8) >> 3),
    (__int64)a3,
    a4);
  return v13;
}
