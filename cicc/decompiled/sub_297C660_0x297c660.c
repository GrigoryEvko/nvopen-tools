// Function: sub_297C660
// Address: 0x297c660
//
__int64 __fastcall sub_297C660(_DWORD *a1, _DWORD *a2)
{
  unsigned int v2; // r8d
  unsigned int v3; // edx
  __int64 result; // rax
  __int64 v5; // [rsp+0h] [rbp-20h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-18h]

  v2 = a1[2];
  v3 = a2[2];
  if ( v2 < v3 )
  {
    sub_C44830((__int64)&v5, a1, v3);
    if ( a1[2] > 0x40u && *(_QWORD *)a1 )
      j_j___libc_free_0_0(*(_QWORD *)a1);
    *(_QWORD *)a1 = v5;
    result = v6;
    a1[2] = v6;
  }
  else if ( v2 > v3 )
  {
    sub_C44830((__int64)&v5, a2, v2);
    if ( a2[2] > 0x40u )
    {
      if ( *(_QWORD *)a2 )
        j_j___libc_free_0_0(*(_QWORD *)a2);
    }
    *(_QWORD *)a2 = v5;
    result = v6;
    a2[2] = v6;
  }
  return result;
}
