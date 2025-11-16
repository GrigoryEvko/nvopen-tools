// Function: sub_2E7D0F0
// Address: 0x2e7d0f0
//
__int64 __fastcall sub_2E7D0F0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 *v8; // rdi
  _QWORD *v9; // rax
  unsigned __int64 v10; // rbx
  unsigned __int64 v12; // rdi
  __int64 v13; // [rsp+0h] [rbp-B0h] BYREF
  char *v14; // [rsp+8h] [rbp-A8h]
  __int64 v15; // [rsp+10h] [rbp-A0h]
  char v16; // [rsp+18h] [rbp-98h] BYREF
  char *v17; // [rsp+20h] [rbp-90h]
  __int64 v18; // [rsp+28h] [rbp-88h]
  char v19; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v20; // [rsp+38h] [rbp-78h]
  __int64 v21; // [rsp+40h] [rbp-70h]
  _BYTE v22[16]; // [rsp+48h] [rbp-68h] BYREF
  __int64 v23; // [rsp+58h] [rbp-58h]
  unsigned __int64 v24; // [rsp+60h] [rbp-50h]
  __int64 v25; // [rsp+68h] [rbp-48h]
  __int64 v26; // [rsp+70h] [rbp-40h]

  v6 = 0xEEEEEEEEEEEEEEEFLL;
  v8 = (__int64 *)a1[55];
  v9 = (_QWORD *)a1[54];
  v10 = 0xEEEEEEEEEEEEEEEFLL * (v8 - v9);
  if ( !(_DWORD)v10 )
  {
LABEL_6:
    v13 = a2;
    v15 = 0x100000000LL;
    v14 = &v16;
    v17 = &v19;
    v18 = 0x100000000LL;
    v20 = v22;
    v21 = 0x100000000LL;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    if ( v8 == (__int64 *)a1[56] )
    {
      sub_2E7CC50(a1 + 54, v8, (__int64)&v13, a4, (__int64)(a1 + 54), a6);
      v12 = v24;
    }
    else
    {
      if ( !v8 )
      {
        a1[55] = 120;
LABEL_13:
        if ( v17 != &v19 )
          _libc_free((unsigned __int64)v17);
        if ( v14 != &v16 )
          _libc_free((unsigned __int64)v14);
        return a1[54] + 120LL * (unsigned int)v10;
      }
      sub_2E7CAB0((__int64)v8, (__int64)&v13, v6, a4, a5, a6);
      v12 = v24;
      a1[55] += 120LL;
    }
    if ( v12 )
      j_j___libc_free_0(v12);
    if ( v20 != v22 )
      _libc_free((unsigned __int64)v20);
    goto LABEL_13;
  }
  a4 = (unsigned int)(v10 - 1);
  v6 = (__int64)&v9[15 * a4 + 15];
  while ( 1 )
  {
    a5 = (__int64)v9;
    if ( *v9 == a2 )
      return a5;
    v9 += 15;
    if ( (_QWORD *)v6 == v9 )
      goto LABEL_6;
  }
}
