// Function: sub_2254030
// Address: 0x2254030
//
_BYTE *__fastcall sub_2254030(_BYTE *a1, float *a2, _DWORD *a3, float a4)
{
  _BYTE *result; // rax
  _BYTE *v6; // [rsp+8h] [rbp-20h]

  __strtof_l();
  result = v6;
  *a2 = a4;
  if ( v6 == a1 || *v6 )
  {
    *a2 = 0.0;
    *a3 = 4;
  }
  else if ( a4 == INFINITY )
  {
    *a2 = 3.4028235e38;
    *a3 = 4;
  }
  else if ( a4 == -INFINITY )
  {
    *a2 = -3.4028235e38;
    *a3 = 4;
  }
  return result;
}
