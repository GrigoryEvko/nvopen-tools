// Function: sub_155F6F0
// Address: 0x155f6f0
//
_QWORD *__fastcall sub_155F6F0(_QWORD *a1, __int64 a2, char a3)
{
  __int64 *v3; // r13
  __int64 v4; // r12
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned __int64 v13; // [rsp+8h] [rbp-78h]
  __int64 v14; // [rsp+10h] [rbp-70h]
  _QWORD v15[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v16; // [rsp+40h] [rbp-40h] BYREF

  v3 = (__int64 *)(a2 + 24);
  *a1 = a1 + 2;
  a1[1] = 0;
  *((_BYTE *)a1 + 16) = 0;
  v4 = a2 + 8LL * *(unsigned int *)(a2 + 16) + 24;
  if ( v4 != a2 + 24 )
  {
    while ( 1 )
    {
      sub_155D8D0((__int64)v15, v3, a3);
      sub_2241490(a1, v15[0], v15[1], v6);
      if ( (__int64 *)v15[0] != &v16 )
        j_j___libc_free_0(v15[0], v16 + 1);
      if ( ++v3 == (__int64 *)v4 )
        break;
      if ( v3 != (__int64 *)(a2 + 24) )
      {
        v7 = a1[1];
        v8 = *a1;
        v9 = v7 + 1;
        if ( a1 + 2 == (_QWORD *)*a1 )
          v10 = 15;
        else
          v10 = a1[2];
        if ( v9 > v10 )
        {
          v13 = v7 + 1;
          v14 = a1[1];
          sub_2240BB0(a1, v14, 0, 0, 1);
          v8 = *a1;
          v9 = v13;
          v7 = v14;
        }
        *(_BYTE *)(v8 + v7) = 32;
        v11 = *a1;
        a1[1] = v9;
        *(_BYTE *)(v11 + v7 + 1) = 0;
      }
    }
  }
  return a1;
}
