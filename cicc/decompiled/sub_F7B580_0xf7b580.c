// Function: sub_F7B580
// Address: 0xf7b580
//
void __fastcall sub_F7B580(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 *v5; // r15
  __int64 v6; // rbx

  if ( (char *)a2 - (char *)a1 <= 224 )
  {
    sub_F7B350(a1, a2, a3);
  }
  else
  {
    v4 = ((char *)a2 - (char *)a1) >> 5;
    v5 = &a1[2 * v4];
    v6 = (16 * v4) >> 4;
    sub_F7B580(a1, v5);
    sub_F7B580(v5, a2);
    sub_F7AB30(a1, v5, (__int64)a2, v6, ((char *)a2 - (char *)v5) >> 4, a3);
  }
}
