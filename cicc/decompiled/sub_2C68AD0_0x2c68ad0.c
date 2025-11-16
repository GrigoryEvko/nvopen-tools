// Function: sub_2C68AD0
// Address: 0x2c68ad0
//
__int64 __fastcall sub_2C68AD0(
        unsigned int *a1,
        unsigned int *a2,
        unsigned int *a3,
        __int64 a4,
        __int64 **a5,
        _BYTE **a6)
{
  __int64 v9; // rax
  unsigned int *v10; // r10
  char *v11; // rdx
  __int64 v12; // r11
  __int64 v14; // [rsp+0h] [rbp-50h]
  __int64 v15; // [rsp+8h] [rbp-48h]
  unsigned int *v19; // [rsp+18h] [rbp-38h]
  unsigned int *v20; // [rsp+18h] [rbp-38h]

  v9 = ((((char *)a2 - (char *)a1) >> 3) + 1) / 2;
  if ( v9 <= a4 )
  {
    v14 = 8 * v9;
    v20 = &a1[2 * v9];
    sub_2C4F4D0(a1, v20, a3, a5, a6);
    sub_2C4F4D0(v20, a2, a3, a5, a6);
    v12 = v14;
    v11 = (char *)a3;
    v10 = v20;
  }
  else
  {
    v15 = 8 * v9;
    v19 = &a1[2 * v9];
    sub_2C68AD0(a1, v19, a3, a4);
    sub_2C68AD0(v19, a2, a3, a4);
    v10 = v19;
    v11 = (char *)a3;
    v12 = v15;
  }
  return sub_2C682C0(a1, v10, (__int64)a2, v12 >> 3, ((char *)a2 - (char *)v10) >> 3, v11, a4, a5, a6);
}
