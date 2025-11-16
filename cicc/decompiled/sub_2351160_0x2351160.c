// Function: sub_2351160
// Address: 0x2351160
//
__int64 __fastcall sub_2351160(unsigned __int64 *a1, __int64 *a2, int a3)
{
  unsigned __int64 v4; // r13
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r12
  __int64 v7; // r13
  _QWORD *v8; // rax
  _QWORD *v9; // r12

  if ( a3 == 1 )
  {
    *a1 = *a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v4 = *a1;
      if ( *a1 )
      {
        v5 = *(unsigned __int64 **)v4;
        v6 = (unsigned __int64 *)(*(_QWORD *)v4 + 32LL * *(unsigned int *)(v4 + 8));
        if ( *(unsigned __int64 **)v4 != v6 )
        {
          do
          {
            v6 -= 4;
            if ( (unsigned __int64 *)*v6 != v6 + 2 )
              j_j___libc_free_0(*v6);
          }
          while ( v5 != v6 );
          v6 = *(unsigned __int64 **)v4;
        }
        if ( v6 != (unsigned __int64 *)(v4 + 16) )
          _libc_free((unsigned __int64)v6);
        j_j___libc_free_0(v4);
      }
    }
    return 0;
  }
  v7 = *a2;
  v8 = (_QWORD *)sub_22077B0(0x10u);
  v9 = v8;
  if ( v8 )
  {
    v8[1] = 0;
    *v8 = v8 + 2;
    if ( *(_DWORD *)(v7 + 8) )
      sub_2350F70((__int64)v8, v7);
  }
  *a1 = (unsigned __int64)v9;
  return 0;
}
