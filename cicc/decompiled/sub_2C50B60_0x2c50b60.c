// Function: sub_2C50B60
// Address: 0x2c50b60
//
void __fastcall sub_2C50B60(unsigned int *a1, unsigned int *a2, __int64 **a3, _BYTE **a4)
{
  __int64 v6; // rax
  unsigned int *v7; // rbx
  __int64 v8; // r9
  __int64 v9; // [rsp+8h] [rbp-38h]

  if ( (char *)a2 - (char *)a1 <= 112 )
  {
    sub_2C4F5D0(a1, a2, a3, a4);
  }
  else
  {
    v6 = 2 * (((char *)a2 - (char *)a1) >> 4);
    v7 = &a1[v6];
    v9 = v6 * 4;
    sub_2C50B60(a1, &a1[v6], a3, a4);
    sub_2C50B60(v7, a2, a3, a4);
    sub_2C50850(a1, v7, (__int64)a2, v9 >> 3, ((char *)a2 - (char *)v7) >> 3, v8, a3, a4);
  }
}
