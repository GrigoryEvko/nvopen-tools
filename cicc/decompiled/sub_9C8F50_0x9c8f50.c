// Function: sub_9C8F50
// Address: 0x9c8f50
//
__int64 __fastcall sub_9C8F50(__int64 *a1)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdi
  __int64 result; // rax
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // rdi

  v2 = a1[1];
  v3 = *a1;
  if ( v2 != *a1 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 + 48);
      v5 = *(_QWORD *)(v3 + 40);
      if ( v4 != v5 )
      {
        do
        {
          if ( *(_DWORD *)(v5 + 40) > 0x40u )
          {
            v6 = *(_QWORD *)(v5 + 32);
            if ( v6 )
              result = j_j___libc_free_0_0(v6);
          }
          if ( *(_DWORD *)(v5 + 24) > 0x40u )
          {
            v8 = *(_QWORD *)(v5 + 16);
            if ( v8 )
              result = j_j___libc_free_0_0(v8);
          }
          v5 += 48;
        }
        while ( v4 != v5 );
        v5 = *(_QWORD *)(v3 + 40);
      }
      if ( v5 )
        result = j_j___libc_free_0(v5, *(_QWORD *)(v3 + 56) - v5);
      if ( *(_DWORD *)(v3 + 32) > 0x40u )
      {
        v9 = *(_QWORD *)(v3 + 24);
        if ( v9 )
          result = j_j___libc_free_0_0(v9);
      }
      if ( *(_DWORD *)(v3 + 16) > 0x40u )
      {
        v10 = *(_QWORD *)(v3 + 8);
        if ( v10 )
          result = j_j___libc_free_0_0(v10);
      }
      v3 += 64;
    }
    while ( v2 != v3 );
    v3 = *a1;
  }
  if ( v3 )
    return j_j___libc_free_0(v3, a1[2] - v3);
  return result;
}
