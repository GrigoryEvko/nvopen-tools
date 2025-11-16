// Function: sub_1111050
// Address: 0x1111050
//
_BOOL8 __fastcall sub_1111050(__int64 *a1, __int64 a2, __int64 *a3, char a4)
{
  __int64 v5; // rdi
  bool v7; // [rsp+Fh] [rbp-21h] BYREF
  __int64 v8; // [rsp+10h] [rbp-20h] BYREF
  int v9; // [rsp+18h] [rbp-18h]

  if ( a4 )
  {
    sub_C45F70((__int64)&v8, a2, (__int64)a3, &v7);
    if ( *((_DWORD *)a1 + 2) > 0x40u )
    {
      v5 = *a1;
      if ( *a1 )
LABEL_4:
        j_j___libc_free_0_0(v5);
    }
  }
  else
  {
    sub_C49AB0((__int64)&v8, a2, a3, &v7);
    if ( *((_DWORD *)a1 + 2) > 0x40u )
    {
      v5 = *a1;
      if ( *a1 )
        goto LABEL_4;
    }
  }
  *a1 = v8;
  *((_DWORD *)a1 + 2) = v9;
  return v7;
}
