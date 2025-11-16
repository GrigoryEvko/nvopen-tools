// Function: sub_C9E0E0
// Address: 0xc9e0e0
//
double *__fastcall sub_C9E0E0(double *a1, char a2)
{
  __int64 v2; // rax
  __int64 v4; // [rsp+8h] [rbp-28h] BYREF
  __int64 v5; // [rsp+10h] [rbp-20h] BYREF
  __suseconds_t v6[2]; // [rsp+18h] [rbp-18h] BYREF

  *a1 = 0.0;
  a1[1] = 0.0;
  a1[2] = 0.0;
  a1[3] = 0.0;
  a1[4] = 0.0;
  v4 = 0;
  if ( a2 )
  {
    if ( !qword_4F84F60 )
      sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
    v2 = 0;
    if ( *(_BYTE *)(qword_4F84F60 + 400) )
      v2 = sub_C86080();
    *((_QWORD *)a1 + 3) = v2;
    sub_C860A0(&v4, &v5, v6);
  }
  else
  {
    sub_C860A0(&v4, &v5, v6);
    if ( !qword_4F84F60 )
      sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
    if ( *(_BYTE *)(qword_4F84F60 + 400) )
      *((_QWORD *)a1 + 3) = sub_C86080();
    else
      a1[3] = 0.0;
  }
  *a1 = (double)(int)v4 / 1000000000.0;
  a1[1] = (double)(int)v5 / 1000000000.0;
  a1[2] = (double)SLODWORD(v6[0]) / 1000000000.0;
  return a1;
}
