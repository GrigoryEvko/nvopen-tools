// Function: sub_3121100
// Address: 0x3121100
//
__int64 __fastcall sub_3121100(unsigned __int64 **a1, __int64 *a2, int a3)
{
  unsigned __int64 *v4; // r12
  __int64 v5; // r13
  unsigned __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 *v11; // r12

  if ( a3 == 1 )
  {
    *a1 = (unsigned __int64 *)*a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v4 = *a1;
      if ( *a1 )
      {
        if ( (unsigned __int64 *)*v4 != v4 + 2 )
          _libc_free(*v4);
        j_j___libc_free_0((unsigned __int64)v4);
      }
    }
    return 0;
  }
  v5 = *a2;
  v6 = (unsigned __int64 *)sub_22077B0(0x60u);
  v11 = v6;
  if ( v6 )
  {
    *v6 = (unsigned __int64)(v6 + 2);
    v6[1] = 0x400000000LL;
    if ( *(_DWORD *)(v5 + 8) )
      sub_3120EC0((__int64)v6, v5, v7, v8, v9, v10);
    v11[6] = *(_QWORD *)(v5 + 48);
    v11[7] = *(_QWORD *)(v5 + 56);
    v11[8] = *(_QWORD *)(v5 + 64);
    v11[9] = *(_QWORD *)(v5 + 72);
    v11[10] = *(_QWORD *)(v5 + 80);
    v11[11] = *(_QWORD *)(v5 + 88);
  }
  *a1 = v11;
  return 0;
}
