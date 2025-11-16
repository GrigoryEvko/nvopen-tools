// Function: sub_31B6640
// Address: 0x31b6640
//
__int64 __fastcall sub_31B6640(__int64 a1, __int64 a2, __int64 a3)
{
  char *v4; // r12
  char *v5; // r14
  __int64 v6; // rsi
  char *v7; // rbx
  unsigned __int64 v8; // r13
  char *v10; // [rsp+0h] [rbp-60h] BYREF
  int v11; // [rsp+8h] [rbp-58h]
  char v12; // [rsp+10h] [rbp-50h] BYREF

  sub_371C6D0(&v10, a2, *(_QWORD *)(a3 + 16));
  v4 = v10;
  v5 = &v10[8 * v11];
  if ( v5 != v10 )
  {
    do
    {
      v6 = *(_QWORD *)v4;
      v4 += 8;
      sub_318D2B0(a1 + 40, v6, a3);
    }
    while ( v5 != v4 );
    v7 = v10;
    v4 = &v10[8 * v11];
    if ( v10 != v4 )
    {
      do
      {
        v8 = *((_QWORD *)v4 - 1);
        v4 -= 8;
        if ( v8 )
        {
          sub_371BB90(v8);
          j_j___libc_free_0(v8);
        }
      }
      while ( v7 != v4 );
      v4 = v10;
    }
  }
  if ( v4 != &v12 )
    _libc_free((unsigned __int64)v4);
  return 0;
}
