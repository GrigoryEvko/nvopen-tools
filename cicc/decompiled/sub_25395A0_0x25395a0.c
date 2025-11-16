// Function: sub_25395A0
// Address: 0x25395a0
//
__int64 __fastcall sub_25395A0(unsigned __int64 *a1, unsigned __int64 *a2, int a3)
{
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  _QWORD *v12; // r12

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
        v5 = *(_QWORD *)(v4 + 16);
        if ( v5 != v4 + 32 )
          _libc_free(v5);
        j_j___libc_free_0(v4);
      }
    }
    return 0;
  }
  v6 = *a2;
  v7 = (_QWORD *)sub_22077B0(0xA0u);
  v12 = v7;
  if ( v7 )
  {
    *v7 = *(_QWORD *)v6;
    v7[1] = *(_QWORD *)(v6 + 8);
    v7[2] = v7 + 4;
    v7[3] = 0x1000000000LL;
    if ( *(_DWORD *)(v6 + 24) )
      sub_2538470((__int64)(v7 + 2), v6 + 16, v8, v9, v10, v11);
  }
  *a1 = (unsigned __int64)v12;
  return 0;
}
