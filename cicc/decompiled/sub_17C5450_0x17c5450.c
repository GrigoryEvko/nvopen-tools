// Function: sub_17C5450
// Address: 0x17c5450
//
__int64 *__fastcall sub_17C5450(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // eax
  __int64 v5; // rax
  const char *v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  __int64 v9; // r13
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD v13[3]; // [rsp+0h] [rbp-F0h] BYREF
  _QWORD *v14; // [rsp+18h] [rbp-D8h] BYREF
  const char *v15; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int64 v16; // [rsp+28h] [rbp-C8h]
  _QWORD v17[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v18; // [rsp+40h] [rbp-B0h]
  _QWORD v19[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v20; // [rsp+60h] [rbp-90h]
  char *v21; // [rsp+70h] [rbp-80h] BYREF
  const char **v22; // [rsp+78h] [rbp-78h]
  __int16 v23; // [rsp+80h] [rbp-70h]
  void *s2; // [rsp+90h] [rbp-60h] BYREF
  size_t n; // [rsp+98h] [rbp-58h]
  _WORD v26[40]; // [rsp+A0h] [rbp-50h] BYREF

  v4 = *(_DWORD *)(a2 + 20);
  v13[0] = a3;
  v13[1] = a4;
  v5 = sub_1649C60(*(_QWORD *)(a2 - 24LL * (v4 & 0xFFFFFFF)));
  v6 = sub_1649960(v5);
  v8 = 0;
  if ( v7 > 7 )
  {
    v8 = v7 - 8;
    v7 = 8;
  }
  v16 = v8;
  v15 = &v6[v7];
  if ( byte_4FA3B40
    && (v9 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL), (unsigned __int8)sub_1695980(*(_QWORD *)(v9 + 40)))
    && (unsigned __int8)sub_1695A10(v9, 0) )
  {
    v11 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v12 = *(_QWORD **)(v11 + 24);
    if ( *(_DWORD *)(v11 + 32) > 0x40u )
      v12 = (_QWORD *)*v12;
    v14 = v12;
    v23 = 2819;
    s2 = v26;
    n = 0x1800000000LL;
    v21 = ".";
    v22 = (const char **)&v14;
    sub_16E2F40((__int64)&v21, (__int64)&s2);
    if ( v16 < (unsigned int)n || (_DWORD)n && memcmp(&v15[v16 - (unsigned int)n], s2, (unsigned int)n) )
    {
      v17[0] = v13;
      v17[1] = &v15;
      v19[0] = v17;
      v23 = 2818;
      v18 = 1285;
      v19[1] = ".";
      v20 = 770;
      v21 = (char *)v19;
      v22 = (const char **)&v14;
      sub_16E2FC0(a1, (__int64)&v21);
    }
    else
    {
      v21 = (char *)v13;
      v22 = &v15;
      v23 = 1285;
      sub_16E2FC0(a1, (__int64)&v21);
    }
    if ( s2 != v26 )
      _libc_free((unsigned __int64)s2);
  }
  else
  {
    s2 = v13;
    n = (size_t)&v15;
    v26[0] = 1285;
    sub_16E2FC0(a1, (__int64)&s2);
  }
  return a1;
}
