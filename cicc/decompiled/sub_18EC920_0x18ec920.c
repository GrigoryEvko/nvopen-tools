// Function: sub_18EC920
// Address: 0x18ec920
//
__int64 __fastcall sub_18EC920(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 i; // r15
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v7; // [rsp+8h] [rbp-58h]
  __int64 v8; // [rsp+10h] [rbp-50h] BYREF
  __int64 v9; // [rsp+18h] [rbp-48h]
  __int64 v10; // [rsp+20h] [rbp-40h]
  int v11; // [rsp+28h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 80);
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v7 = v2;
  if ( v2 == a2 + 72 )
  {
    v5 = 0;
  }
  else
  {
    do
    {
      if ( !v7 )
        BUG();
      for ( i = *(_QWORD *)(v7 + 24); v7 + 16 != i; i = *(_QWORD *)(i + 8) )
      {
        v4 = i - 24;
        if ( !i )
          v4 = 0;
        sub_18EC860(a1, (__int64)&v8, v4);
      }
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( a2 + 72 != v7 );
    v5 = v9;
  }
  return j___libc_free_0(v5);
}
