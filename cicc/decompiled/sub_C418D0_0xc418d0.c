// Function: sub_C418D0
// Address: 0xc418d0
//
_QWORD *__fastcall sub_C418D0(_QWORD *a1, _DWORD *a2)
{
  unsigned int v2; // eax
  unsigned __int64 v3; // rdx
  unsigned __int64 v5; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-28h]

  v2 = a2[3];
  v6 = v2;
  if ( v2 > 0x40 )
  {
    sub_C43690(&v5, -1, 1);
  }
  else
  {
    v3 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    if ( !v2 )
      v3 = 0;
    v5 = v3;
  }
  if ( a2 == sub_C33340() )
    sub_C3C640(a1, (__int64)a2, &v5);
  else
    sub_C3B160((__int64)a1, a2, (__int64 *)&v5);
  if ( v6 > 0x40 && v5 )
    j_j___libc_free_0_0(v5);
  return a1;
}
