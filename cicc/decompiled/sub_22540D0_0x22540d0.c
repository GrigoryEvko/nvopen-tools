// Function: sub_22540D0
// Address: 0x22540d0
//
__int64 __fastcall sub_22540D0(_BYTE *a1, double *a2, _DWORD *a3, double a4)
{
  __int64 result; // rax
  _BYTE *v6; // [rsp+8h] [rbp-20h]

  __strtod_l();
  result = (__int64)v6;
  *a2 = a4;
  if ( v6 == a1 || *v6 )
  {
    *a2 = 0.0;
    *a3 = 4;
  }
  else if ( a4 == INFINITY )
  {
    *a2 = 1.797693134862316e308;
    *a3 = 4;
    return 0x7FEFFFFFFFFFFFFFLL;
  }
  else if ( a4 == -INFINITY )
  {
    *a2 = -1.797693134862316e308;
    *a3 = 4;
    return 0xFFEFFFFFFFFFFFFFLL;
  }
  return result;
}
