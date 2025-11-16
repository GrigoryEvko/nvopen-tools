// Function: sub_1C1FB60
// Address: 0x1c1fb60
//
__int64 __fastcall sub_1C1FB60(__int64 a1, __int64 *a2)
{
  int v2; // ebx
  __int64 v3; // r13
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  const char *v11; // rax
  __int64 v12; // rdx
  void (__fastcall *v13)(__int64, const char ***); // rax
  __int64 v15; // rax
  unsigned __int8 *v16; // [rsp+28h] [rbp-A8h]
  __int64 v17; // [rsp+38h] [rbp-98h] BYREF
  __int64 v18; // [rsp+40h] [rbp-90h] BYREF
  __int64 v19; // [rsp+48h] [rbp-88h]
  const char *v20; // [rsp+50h] [rbp-80h] BYREF
  __int64 v21; // [rsp+58h] [rbp-78h]
  _QWORD v22[2]; // [rsp+60h] [rbp-70h] BYREF
  const char **v23; // [rsp+70h] [rbp-60h] BYREF
  __int64 v24; // [rsp+78h] [rbp-58h]
  __int64 v25; // [rsp+80h] [rbp-50h]
  __int64 v26; // [rsp+88h] [rbp-48h]
  int v27; // [rsp+90h] [rbp-40h]
  __int64 *v28; // [rsp+98h] [rbp-38h]

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 64LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v2 = a2[1] - *(_DWORD *)a2;
  if ( v2 )
  {
    v3 = (unsigned int)(v2 - 1);
    v4 = 1;
    v5 = 0;
    v6 = v3 + 2;
    do
    {
      while ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 72LL))(
                 a1,
                 (unsigned int)v5,
                 &v17) )
      {
        ++v4;
        ++v5;
        if ( v6 == v4 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
      }
      v7 = *a2;
      v8 = a2[1] - *a2;
      if ( v8 <= v5 )
      {
        if ( v8 < v4 )
        {
          sub_CD93F0(a2, v4 - v8);
          v7 = *a2;
        }
        else if ( v8 > v4 && a2[1] != v7 + v4 )
        {
          a2[1] = v7 + v4;
        }
      }
      v16 = (unsigned __int8 *)(v5 + v7);
      if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
      {
        LOBYTE(v22[0]) = 0;
        v20 = (const char *)v22;
        v21 = 0;
        v23 = (const char **)&unk_49EFBE0;
        v27 = 1;
        v26 = 0;
        v25 = 0;
        v24 = 0;
        v28 = (__int64 *)&v20;
        v15 = sub_16E4080(a1);
        sub_16E5960(v16, v15, (__int64)&v23);
        if ( v26 != v24 )
          sub_16E7BA0((__int64 *)&v23);
        v18 = *v28;
        v19 = v28[1];
        (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(*(_QWORD *)a1 + 216LL))(a1, &v18, 0);
        sub_16E7BC0((__int64 *)&v23);
        if ( v20 != (const char *)v22 )
          j_j___libc_free_0(v20, v22[0] + 1LL);
      }
      else
      {
        v9 = *(_QWORD *)a1;
        v18 = 0;
        v19 = 0;
        (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(v9 + 216))(a1, &v18, 0);
        v10 = sub_16E4080(a1);
        v11 = sub_16E5970(v18, v19, v10, v16);
        v21 = v12;
        v20 = v11;
        if ( v12 )
        {
          v13 = *(void (__fastcall **)(__int64, const char ***))(*(_QWORD *)a1 + 232LL);
          LOWORD(v25) = 261;
          v23 = &v20;
          v13(a1, &v23);
        }
      }
      ++v4;
      ++v5;
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 80LL))(a1, v17);
    }
    while ( v6 != v4 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
}
