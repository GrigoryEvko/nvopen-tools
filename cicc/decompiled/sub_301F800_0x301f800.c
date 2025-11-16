// Function: sub_301F800
// Address: 0x301f800
//
void __fastcall sub_301F800(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 *v4; // rdi
  unsigned __int64 *v5; // r13
  unsigned __int64 *v6; // rbx
  _QWORD v7[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v8; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 16);
  v3 = v2 + 32LL * *(unsigned int *)(a1 + 24);
  if ( v3 != v2 )
  {
    do
    {
      v4 = *(__int64 **)(a1 + 8);
      v7[0] = v2;
      v2 += 32;
      v8 = 260;
      sub_E99A90(v4, (__int64)v7);
    }
    while ( v3 != v2 );
    v5 = *(unsigned __int64 **)(a1 + 16);
    v6 = &v5[4 * *(unsigned int *)(a1 + 24)];
    while ( v5 != v6 )
    {
      while ( 1 )
      {
        v6 -= 4;
        if ( (unsigned __int64 *)*v6 == v6 + 2 )
          break;
        j_j___libc_free_0(*v6);
        if ( v5 == v6 )
          goto LABEL_7;
      }
    }
  }
LABEL_7:
  *(_DWORD *)(a1 + 24) = 0;
}
