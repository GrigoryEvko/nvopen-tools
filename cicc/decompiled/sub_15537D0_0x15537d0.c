// Function: sub_15537D0
// Address: 0x15537d0
//
char __fastcall sub_15537D0(__int64 a1, __int64 a2, char a3, _QWORD *a4)
{
  _QWORD *v4; // r12
  char result; // al
  unsigned __int64 v7; // r8
  __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  __int64 v11[6]; // [rsp+10h] [rbp-170h] BYREF
  _BYTE v12[40]; // [rsp+40h] [rbp-140h] BYREF
  __int64 v13; // [rsp+68h] [rbp-118h]
  __int64 v14; // [rsp+90h] [rbp-F0h]
  __int64 v15; // [rsp+B8h] [rbp-C8h]
  __int64 v16; // [rsp+E0h] [rbp-A0h]
  unsigned __int64 v17; // [rsp+100h] [rbp-80h]
  unsigned int v18; // [rsp+108h] [rbp-78h]
  int v19; // [rsp+10Ch] [rbp-74h]
  __int64 v20; // [rsp+130h] [rbp-50h]

  v4 = a4;
  if ( !a4 )
    v4 = sub_1548BC0(a1);
  if ( a3 || (result = sub_1553590(a1, a2, 0, (__int64)v4)) == 0 )
  {
    sub_154BB30((__int64)v12, (__int64)v4, *(_BYTE *)(a1 + 16) == 19);
    sub_154B9F0((__int64)v11, (__int64)v12, (__int64)v4, 0);
    sub_15535E0((__int64 *)a1, a2, a3, (__int64)v11);
    sub_154BA40(v11);
    j___libc_free_0(v20);
    if ( v19 )
    {
      v7 = v17;
      if ( v18 )
      {
        v8 = 8LL * v18;
        v9 = 0;
        do
        {
          v10 = *(_QWORD *)(v7 + v9);
          if ( v10 )
          {
            if ( v10 != -8 )
            {
              _libc_free(v10);
              v7 = v17;
            }
          }
          v9 += 8;
        }
        while ( v8 != v9 );
      }
    }
    else
    {
      v7 = v17;
    }
    _libc_free(v7);
    j___libc_free_0(v16);
    j___libc_free_0(v15);
    j___libc_free_0(v14);
    return j___libc_free_0(v13);
  }
  return result;
}
