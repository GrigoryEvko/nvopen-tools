// Function: sub_16D7810
// Address: 0x16d7810
//
double *__fastcall sub_16D7810(double *a1, char a2)
{
  __int64 v2; // rax
  __int64 v4; // [rsp+8h] [rbp-28h] BYREF
  __int64 v5; // [rsp+10h] [rbp-20h] BYREF
  __suseconds_t v6[2]; // [rsp+18h] [rbp-18h] BYREF

  *a1 = 0.0;
  a1[1] = 0.0;
  a1[2] = 0.0;
  a1[3] = 0.0;
  v4 = 0;
  if ( a2 )
  {
    v2 = 0;
    if ( byte_4FA15E0 )
      v2 = sub_16C68E0();
    *((_QWORD *)a1 + 3) = v2;
    sub_16C6900(&v4, &v5, v6);
  }
  else
  {
    sub_16C6900(&v4, &v5, v6);
    if ( byte_4FA15E0 )
      *((_QWORD *)a1 + 3) = sub_16C68E0();
    else
      a1[3] = 0.0;
  }
  *a1 = (double)(int)v4 / 1000000000.0;
  a1[1] = (double)(int)v5 / 1000000000.0;
  a1[2] = (double)SLODWORD(v6[0]) / 1000000000.0;
  return a1;
}
