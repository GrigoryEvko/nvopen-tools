// Function: sub_2EAC1F0
// Address: 0x2eac1f0
//
char __fastcall sub_2EAC1F0(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  char result; // al
  unsigned __int8 *v5; // r13
  __int64 v6; // rbx
  __int64 v8; // r12
  char v9; // [rsp+Fh] [rbp-41h]
  unsigned __int64 v10; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-38h]

  result = 0;
  if ( ((*a1 >> 2) & 1) == 0 )
  {
    v5 = (unsigned __int8 *)(*a1 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v5 )
    {
      v6 = (*a1 >> 2) & 1;
      if ( *v5 >= 0x1Du )
        v6 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
      v8 = a1[1] + a2;
      v11 = sub_AE2980(a4, 0)[1];
      if ( v11 > 0x40 )
        sub_C43690((__int64)&v10, v8, 0);
      else
        v10 = v8;
      result = sub_D30550(v5, 0, &v10, a4, v6, 0, 0, 0);
      if ( v11 > 0x40 )
      {
        if ( v10 )
        {
          v9 = result;
          j_j___libc_free_0_0(v10);
          return v9;
        }
      }
    }
  }
  return result;
}
