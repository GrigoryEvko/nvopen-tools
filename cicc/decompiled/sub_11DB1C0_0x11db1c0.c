// Function: sub_11DB1C0
// Address: 0x11db1c0
//
__int64 __fastcall sub_11DB1C0(__int64 a1)
{
  char v1; // al
  __int64 v2; // r13
  __int64 *v4; // rsi
  void *v5; // rbx
  _DWORD *v6; // rax
  __int64 *v7; // rax
  __int64 v8; // r13
  _QWORD *i; // r12
  bool v10; // [rsp+Fh] [rbp-41h] BYREF
  void *v11; // [rsp+10h] [rbp-40h] BYREF
  _QWORD *v12; // [rsp+18h] [rbp-38h]

  v1 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 <= 0x1Cu )
  {
    if ( v1 != 18 )
      return 0;
    v4 = (__int64 *)(a1 + 24);
    v5 = sub_C33340();
    if ( *(void **)(a1 + 24) == v5 )
      sub_C3C790(&v11, (_QWORD **)v4);
    else
      sub_C33EB0(&v11, v4);
    v6 = sub_C33310();
    sub_C41640((__int64 *)&v11, v6, 1, &v10);
    if ( v10 )
    {
      sub_91D830(&v11);
      return 0;
    }
    else
    {
      v7 = (__int64 *)sub_BD5C60(a1);
      v2 = sub_AC8EA0(v7, (__int64 *)&v11);
      if ( v5 == v11 )
      {
        if ( v12 )
        {
          for ( i = &v12[3 * *(v12 - 1)]; v12 != i; sub_91D830(i) )
            i -= 3;
          j_j_j___libc_free_0_0(i - 1);
        }
        return v2;
      }
      sub_C338F0((__int64)&v11);
      return v2;
    }
  }
  else
  {
    v2 = 0;
    if ( v1 != 75 )
      return v2;
    v8 = *(_QWORD *)(a1 - 32);
    if ( *(_BYTE *)(*(_QWORD *)(v8 + 8) + 8LL) != 2 )
      return 0;
    return v8;
  }
}
