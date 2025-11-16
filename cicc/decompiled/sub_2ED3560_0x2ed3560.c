// Function: sub_2ED3560
// Address: 0x2ed3560
//
void __fastcall sub_2ED3560(__int64 *src, __int64 *a2, __int64 *a3, __int64 *a4, _QWORD *a5)
{
  __int64 *v8; // rbx
  __int64 *v9; // rdi
  __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // [rsp+0h] [rbp-50h]
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 *v14; // [rsp+10h] [rbp-40h]

  v12 = (char *)a2 - (char *)src;
  v13 = a2 - src;
  v14 = (__int64 *)((char *)a3 + (char *)a2 - (char *)src);
  if ( (char *)a2 - (char *)src <= 48 )
  {
    sub_2ED3280(src, a2, (__int64)a4, a5);
  }
  else
  {
    v8 = src;
    do
    {
      v9 = v8;
      v8 += 7;
      sub_2ED3280(v9, v8, (__int64)a4, a5);
    }
    while ( (char *)a2 - (char *)v8 > 48 );
    sub_2ED3280(v8, a2, (__int64)a4, a5);
    if ( v12 > 56 )
    {
      v10 = 7;
      do
      {
        sub_2ED31C0(src, a2, a3, v10, a4, a5);
        v11 = 2 * v10;
        v10 *= 4;
        sub_2ED31C0(a3, v14, src, v11, a4, a5);
      }
      while ( v13 > v10 );
    }
  }
}
