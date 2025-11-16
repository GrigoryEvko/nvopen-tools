// Function: sub_13EE900
// Address: 0x13ee900
//
int *__fastcall sub_13EE900(int *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 v8; // rdi
  __int64 v9; // rdi
  __int64 v10; // [rsp+0h] [rbp-40h] BYREF
  __int64 v11; // [rsp+8h] [rbp-38h]
  __int64 v12; // [rsp+10h] [rbp-30h]
  unsigned int v13; // [rsp+18h] [rbp-28h]

  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  sub_13EDBA0(a1, a2, a3, a4, (__int64)&v10);
  if ( v13 )
  {
    v5 = v11;
    v6 = v11 + 48LL * v13;
    do
    {
      if ( *(_QWORD *)v5 != -16 && *(_QWORD *)v5 != -8 && *(_DWORD *)(v5 + 8) == 3 )
      {
        if ( *(_DWORD *)(v5 + 40) > 0x40u )
        {
          v8 = *(_QWORD *)(v5 + 32);
          if ( v8 )
            j_j___libc_free_0_0(v8);
        }
        if ( *(_DWORD *)(v5 + 24) > 0x40u )
        {
          v9 = *(_QWORD *)(v5 + 16);
          if ( v9 )
            j_j___libc_free_0_0(v9);
        }
      }
      v5 += 48;
    }
    while ( v6 != v5 );
  }
  j___libc_free_0(v11);
  return a1;
}
