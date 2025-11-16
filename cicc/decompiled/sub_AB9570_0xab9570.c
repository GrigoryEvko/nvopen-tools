// Function: sub_AB9570
// Address: 0xab9570
//
__int64 __fastcall sub_AB9570(__int64 a1, __int64 a2, __int64 *a3, unsigned int a4, unsigned int a5)
{
  __int64 v9[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v10; // [rsp+20h] [rbp-60h] BYREF
  __int64 v11[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v12[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( sub_AAF7D0(a2) || sub_AAF7D0((__int64)a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else if ( a4 == 2 )
  {
    sub_AB3BD0(a1, a2, (__int64)a3);
  }
  else if ( a4 > 2 )
  {
    if ( a4 != 3 )
      BUG();
    sub_AB3BD0((__int64)v9, a2, (__int64)a3);
    sub_AB0E00((__int64)v11, a2, (__int64)a3);
    sub_AB2160(a1, (__int64)v9, (__int64)v11, a5);
    sub_969240(v12);
    sub_969240(v11);
    sub_969240(&v10);
    sub_969240(v9);
  }
  else if ( a4 )
  {
    sub_AB0E00(a1, a2, (__int64)a3);
  }
  else
  {
    sub_AB8FB0(a1, a2, a3);
  }
  return a1;
}
