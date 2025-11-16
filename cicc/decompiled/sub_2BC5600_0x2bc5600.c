// Function: sub_2BC5600
// Address: 0x2bc5600
//
__int64 __fastcall sub_2BC5600(
        char *src,
        char *a2,
        char *a3,
        __int64 a4,
        unsigned __int8 (__fastcall *a5)(__int64, _QWORD, _QWORD),
        __int64 a6)
{
  __int64 v6; // rax
  char *v9; // r13
  __int64 v10; // rbx
  char *v11; // rdi
  __int64 v13; // [rsp-10h] [rbp-60h]
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v6 = (a2 - src) >> 3;
  v9 = src;
  v16 = 2 * a4;
  if ( 2 * a4 <= v6 )
  {
    v15 = 16 * a4;
    v10 = -8 * a4;
    do
    {
      v11 = v9;
      v9 += v15;
      a3 = sub_2BC5520(v11, &v9[v10], &v9[v10], v9, a3, a6, a5, a6);
      v6 = (a2 - v9) >> 3;
    }
    while ( v6 >= v16 );
  }
  if ( v6 > a4 )
    v6 = a4;
  sub_2BC5520(v9, &v9[8 * v6], &v9[8 * v6], a2, a3, a6, a5, a6);
  return v13;
}
