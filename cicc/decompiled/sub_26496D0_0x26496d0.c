// Function: sub_26496D0
// Address: 0x26496d0
//
void __fastcall sub_26496D0(char *a1, char *a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 *v5; // r15
  __int64 v6; // rbx

  if ( a2 - a1 <= 224 )
  {
    sub_2649280(a1, a2, a3);
  }
  else
  {
    v4 = (a2 - a1) >> 5;
    v5 = (__int64 *)&a1[16 * v4];
    v6 = (16 * v4) >> 4;
    sub_26496D0(a1, v5);
    sub_26496D0(v5, a2);
    sub_26487A0((__int64 *)a1, v5, (__int64)a2, v6, (a2 - (char *)v5) >> 4, a3);
  }
}
