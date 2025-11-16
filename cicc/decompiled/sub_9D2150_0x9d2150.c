// Function: sub_9D2150
// Address: 0x9d2150
//
__int64 __fastcall sub_9D2150(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  __int64 v4; // rdi

  result = *((unsigned __int8 *)a1 + 32);
  if ( (result & 2) != 0 )
    sub_9D20E0(a1);
  if ( (result & 1) != 0 )
  {
    v4 = *a1;
    if ( v4 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  }
  else
  {
    if ( *((_DWORD *)a1 + 6) > 0x40u )
    {
      v3 = a1[2];
      if ( v3 )
        result = j_j___libc_free_0_0(v3);
    }
    if ( *((_DWORD *)a1 + 2) > 0x40u )
    {
      if ( *a1 )
        return j_j___libc_free_0_0(*a1);
    }
  }
  return result;
}
