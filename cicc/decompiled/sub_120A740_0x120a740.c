// Function: sub_120A740
// Address: 0x120a740
//
__int64 __fastcall sub_120A740(__int64 a1)
{
  __int64 v2; // rdi
  void *v3; // r13
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 result; // rax
  void **v8; // rax
  void **v9; // r12

  v2 = *(_QWORD *)(a1 + 144);
  if ( v2 )
    j_j___libc_free_0_0(v2);
  v3 = sub_C33340();
  if ( *(void **)(a1 + 112) == v3 )
  {
    v8 = *(void ***)(a1 + 120);
    if ( v8 )
    {
      v9 = &v8[3 * (_QWORD)*(v8 - 1)];
      if ( v8 != v9 )
      {
        do
        {
          while ( 1 )
          {
            v9 -= 3;
            if ( v3 == *v9 )
              break;
            sub_C338F0((__int64)v9);
            if ( *(void ***)(a1 + 120) == v9 )
              goto LABEL_15;
          }
          sub_969EE0((__int64)v9);
        }
        while ( *(void ***)(a1 + 120) != v9 );
      }
LABEL_15:
      j_j_j___libc_free_0_0(v9 - 1);
    }
  }
  else
  {
    sub_C338F0(a1 + 112);
  }
  if ( *(_DWORD *)(a1 + 104) > 0x40u )
  {
    v4 = *(_QWORD *)(a1 + 96);
    if ( v4 )
      j_j___libc_free_0_0(v4);
  }
  v5 = *(_QWORD *)(a1 + 64);
  if ( v5 != a1 + 80 )
    j_j___libc_free_0(v5, *(_QWORD *)(a1 + 80) + 1LL);
  v6 = *(_QWORD *)(a1 + 32);
  result = a1 + 48;
  if ( v6 != a1 + 48 )
    return j_j___libc_free_0(v6, *(_QWORD *)(a1 + 48) + 1LL);
  return result;
}
