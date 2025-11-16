// Function: sub_2451270
// Address: 0x2451270
//
__int64 *__fastcall sub_2451270(__int64 *a1, __int64 a2, void *a3, void *a4, _BYTE *a5)
{
  unsigned __int8 *v8; // rax
  const char *v9; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r8
  const char *v12; // rcx
  char v13; // al
  char v15; // al
  __int64 v16; // rax
  bool v17; // cc
  _QWORD *v18; // rax
  size_t v19; // r8
  const char *v20; // rcx
  int v21; // eax
  const char *v22; // [rsp+8h] [rbp-118h]
  size_t v23; // [rsp+10h] [rbp-110h]
  __int64 v24; // [rsp+18h] [rbp-108h]
  _QWORD *v25; // [rsp+28h] [rbp-F8h] BYREF
  _QWORD v26[4]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v27; // [rsp+50h] [rbp-D0h]
  _QWORD v28[4]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v29; // [rsp+80h] [rbp-A0h]
  void *v30[2]; // [rsp+90h] [rbp-90h] BYREF
  const char *v31; // [rsp+A0h] [rbp-80h]
  size_t v32; // [rsp+A8h] [rbp-78h]
  __int16 v33; // [rsp+B0h] [rbp-70h]
  void *s2; // [rsp+C0h] [rbp-60h] BYREF
  size_t n; // [rsp+C8h] [rbp-58h]
  __int64 v36; // [rsp+D0h] [rbp-50h]
  unsigned __int64 v37; // [rsp+D8h] [rbp-48h] BYREF
  __int16 v38; // [rsp+E0h] [rbp-40h]

  v8 = sub_BD3990(*(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), a2);
  v9 = sub_BD5D20((__int64)v8);
  v11 = 0;
  if ( v10 > 7 )
  {
    v11 = v10 - 8;
    v10 = 8;
  }
  v12 = &v9[v10];
  if ( !(_BYTE)qword_4FE7388 )
    goto LABEL_5;
  v22 = &v9[v10];
  v23 = v11;
  v24 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL);
  v13 = sub_ED2B90(*(_QWORD *)(v24 + 40));
  v11 = v23;
  v12 = v22;
  if ( !v13 )
    goto LABEL_5;
  v15 = sub_ED2C10(v24, 0);
  v11 = v23;
  v12 = v22;
  if ( v15 )
  {
    *a5 = 1;
    v16 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    v17 = *(_DWORD *)(v16 + 32) <= 0x40u;
    v18 = *(_QWORD **)(v16 + 24);
    if ( !v17 )
      v18 = (_QWORD *)*v18;
    v33 = 2819;
    v31 = (const char *)&v25;
    v30[0] = ".";
    v25 = v18;
    s2 = &v37;
    n = 0;
    v36 = 24;
    sub_CA0EC0((__int64)v30, (__int64)&s2);
    v19 = v23;
    v20 = v22;
    if ( v23 < n || n && (v21 = memcmp(&v22[v23 - n], s2, n), v20 = v22, v19 = v23, v21) )
    {
      v26[2] = v20;
      v28[0] = v26;
      v33 = 2818;
      v26[0] = a3;
      v26[1] = a4;
      v26[3] = v19;
      v27 = 1285;
      v28[2] = ".";
      v29 = 770;
      v30[0] = v28;
      v31 = (const char *)&v25;
      sub_CA0F50(a1, v30);
    }
    else
    {
      v31 = v20;
      v30[0] = a3;
      v30[1] = a4;
      v32 = v19;
      v33 = 1285;
      sub_CA0F50(a1, v30);
    }
    if ( s2 != &v37 )
      _libc_free((unsigned __int64)s2);
  }
  else
  {
LABEL_5:
    *a5 = 0;
    v37 = v11;
    s2 = a3;
    n = (size_t)a4;
    v36 = (__int64)v12;
    v38 = 1285;
    sub_CA0F50(a1, &s2);
  }
  return a1;
}
