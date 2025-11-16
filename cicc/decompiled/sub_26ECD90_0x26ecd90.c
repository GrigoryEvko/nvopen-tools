// Function: sub_26ECD90
// Address: 0x26ecd90
//
void __fastcall sub_26ECD90(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // rsi
  _QWORD *v6; // rbx
  unsigned __int64 v7; // rdi
  void *s; // [rsp+10h] [rbp-70h] BYREF
  __int64 v9; // [rsp+18h] [rbp-68h]
  _QWORD *v10; // [rsp+20h] [rbp-60h]
  __int64 v11; // [rsp+28h] [rbp-58h]
  int v12; // [rsp+30h] [rbp-50h]
  __int64 v13; // [rsp+38h] [rbp-48h]
  _QWORD v14[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( sub_26EB8E0(a1, a2) )
  {
    v3 = *(_QWORD *)(a2 + 80);
    v4 = a2 + 72;
    v9 = 1;
    s = v14;
    v10 = 0;
    v11 = 0;
    v12 = 1065353216;
    v13 = 0;
    for ( v14[0] = 0; v4 != v3; v3 = *(_QWORD *)(v3 + 8) )
    {
      v5 = v3 - 24;
      if ( !v3 )
        v5 = 0;
      sub_26EC2C0(a1, v5, (unsigned __int64 *)&s);
    }
    sub_26EC840(a1, a2, (__int64)&s);
    v6 = v10;
    while ( v6 )
    {
      v7 = (unsigned __int64)v6;
      v6 = (_QWORD *)*v6;
      j_j___libc_free_0(v7);
    }
    memset(s, 0, 8 * v9);
    v11 = 0;
    v10 = 0;
    if ( s != v14 )
      j_j___libc_free_0((unsigned __int64)s);
  }
}
