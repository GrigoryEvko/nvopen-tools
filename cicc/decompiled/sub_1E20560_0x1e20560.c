// Function: sub_1E20560
// Address: 0x1e20560
//
__int64 __fastcall sub_1E20560(__int64 a1, __int64 a2, unsigned __int8 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int *v6; // rdx
  unsigned int *v7; // rsi
  __int64 v8; // rcx
  unsigned int *v9; // rax
  int *v10; // rsi
  int v11; // ecx
  char v13[8]; // [rsp+0h] [rbp-30h] BYREF
  unsigned int *v14; // [rsp+8h] [rbp-28h]
  int v15; // [rsp+10h] [rbp-20h]
  int v16; // [rsp+18h] [rbp-18h]

  sub_1E1F460((__int64)v13, a1, a2, 1, a3, a6);
  if ( v15 )
  {
    v6 = &v14[2 * v16];
    if ( v14 != v6 )
    {
      v7 = v14;
      while ( 1 )
      {
        v8 = *v7;
        v9 = v7;
        if ( (unsigned int)v8 <= 0xFFFFFFFD )
          break;
        v7 += 2;
        if ( v6 == v7 )
          return j___libc_free_0(v14);
      }
      if ( v6 != v7 )
      {
        while ( 1 )
        {
          v10 = (int *)(*(_QWORD *)(a1 + 936) + 4 * v8);
          v11 = *v10 + v9[1];
          if ( *v10 < (signed int)-v9[1] )
            v11 = 0;
          v9 += 2;
          *v10 = v11;
          if ( v9 == v6 )
            break;
          while ( *v9 > 0xFFFFFFFD )
          {
            v9 += 2;
            if ( v6 == v9 )
              return j___libc_free_0(v14);
          }
          if ( v9 == v6 )
            break;
          v8 = *v9;
        }
      }
    }
  }
  return j___libc_free_0(v14);
}
