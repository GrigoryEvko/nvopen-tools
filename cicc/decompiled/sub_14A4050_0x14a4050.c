// Function: sub_14A4050
// Address: 0x14a4050
//
__int64 __fastcall sub_14A4050(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 *v4; // rdi
  _QWORD **v5; // r13
  _QWORD **i; // r12
  __int64 v7; // rax
  _QWORD *v8; // rbx
  _QWORD *v9; // r15
  __int64 v10; // rdi
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  __int64 v13; // rdi
  __int64 v15[2]; // [rsp+8h] [rbp-A8h] BYREF
  _QWORD *v16; // [rsp+18h] [rbp-98h]
  __int64 v17; // [rsp+20h] [rbp-90h]
  __int64 v18; // [rsp+28h] [rbp-88h]
  __int64 v19; // [rsp+30h] [rbp-80h]
  __int64 v20; // [rsp+38h] [rbp-78h]
  __int64 v21; // [rsp+40h] [rbp-70h]
  __int64 v22; // [rsp+48h] [rbp-68h]
  __int64 v23; // [rsp+50h] [rbp-60h]
  __int64 v24; // [rsp+58h] [rbp-58h]
  __int64 v25; // [rsp+60h] [rbp-50h]
  __int64 v26; // [rsp+68h] [rbp-48h]
  char v27; // [rsp+70h] [rbp-40h]

  v3 = a1 + 192;
  v19 = 0;
  v15[1] = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  sub_14A3D40((__int64)v15, a1 + 160, a2);
  v4 = (__int64 *)(a1 + 192);
  if ( *(_BYTE *)(a1 + 200) )
  {
    sub_14A3C60(v4, v15);
  }
  else
  {
    sub_14A26A0(v4, v15);
    *(_BYTE *)(a1 + 200) = 1;
  }
  sub_14A3B20(v15);
  j___libc_free_0(v24);
  if ( (_DWORD)v22 )
  {
    v5 = (_QWORD **)(v20 + 32LL * (unsigned int)v22);
    for ( i = (_QWORD **)(v20 + 8); ; i += 4 )
    {
      v7 = (__int64)*(i - 1);
      if ( v7 != -8 && v7 != -16 )
      {
        v8 = *i;
        while ( v8 != i )
        {
          v9 = v8;
          v8 = (_QWORD *)*v8;
          v10 = v9[3];
          if ( v10 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
          j_j___libc_free_0(v9, 32);
        }
      }
      if ( v5 == i + 3 )
        break;
    }
  }
  j___libc_free_0(v20);
  if ( (_DWORD)v18 )
  {
    v11 = v16;
    v12 = &v16[2 * (unsigned int)v18];
    do
    {
      if ( *v11 != -8 && *v11 != -16 )
      {
        v13 = v11[1];
        if ( v13 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v13 + 8LL))(v13);
      }
      v11 += 2;
    }
    while ( v12 != v11 );
  }
  j___libc_free_0(v16);
  return v3;
}
