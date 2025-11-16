// Function: sub_CB7680
// Address: 0xcb7680
//
__int64 __fastcall sub_CB7680(__int64 a1)
{
  const char *v1; // rax
  __int64 result; // rax
  const char *v3; // rcx

  v1 = *(const char **)a1;
  if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) <= 0 )
  {
    if ( !*(_DWORD *)(a1 + 16) )
      *(_DWORD *)(a1 + 16) = 7;
    *(_QWORD *)(a1 + 8) = byte_4F85140;
    *(_QWORD *)a1 = &byte_4F85140[1];
    return byte_4F85140[0];
  }
  else if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 != 1 && *v1 == 91 && v1[1] == 46 )
  {
    *(_QWORD *)a1 = v1 + 2;
    result = sub_CB7550((const char **)a1, 46);
    v3 = *(const char **)a1;
    if ( (__int64)(*(_QWORD *)(a1 + 8) - *(_QWORD *)a1) > 1 && *v3 == 46 && v3[1] == 93 )
    {
      *(_QWORD *)a1 = v3 + 2;
    }
    else
    {
      if ( !*(_DWORD *)(a1 + 16) )
        *(_DWORD *)(a1 + 16) = 3;
      *(_QWORD *)a1 = byte_4F85140;
      *(_QWORD *)(a1 + 8) = byte_4F85140;
    }
  }
  else
  {
    *(_QWORD *)a1 = v1 + 1;
    return *(unsigned __int8 *)v1;
  }
  return result;
}
