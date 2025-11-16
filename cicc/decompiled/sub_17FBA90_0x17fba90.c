// Function: sub_17FBA90
// Address: 0x17fba90
//
bool __fastcall sub_17FBA90(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  const char *v4; // rax
  unsigned __int64 v5; // rdx
  const char *v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v10; // r14
  size_t v11; // rdx
  size_t v12; // r13
  int v13; // edx
  _QWORD *v14; // r15
  int v15; // [rsp+Ch] [rbp-94h]
  __int64 v16; // [rsp+10h] [rbp-90h] BYREF
  __int16 v17; // [rsp+20h] [rbp-80h]
  void *s2; // [rsp+30h] [rbp-70h] BYREF
  size_t n; // [rsp+38h] [rbp-68h]
  _QWORD v20[4]; // [rsp+40h] [rbp-60h] BYREF
  int v21; // [rsp+64h] [rbp-3Ch]

  v2 = sub_164A820(a2);
  v3 = v2;
  if ( *(_BYTE *)(v2 + 16) != 3 )
  {
LABEL_8:
    v8 = *(_QWORD *)v3;
    if ( *(_BYTE *)(*(_QWORD *)v3 + 8LL) == 16 )
    {
      v8 = **(_QWORD **)(v8 + 16);
      if ( *(_BYTE *)(v8 + 8) == 16 )
        v8 = **(_QWORD **)(v8 + 16);
    }
    return *(_DWORD *)(v8 + 8) >> 8 == 0;
  }
  if ( (*(_BYTE *)(v2 + 34) & 0x20) == 0 )
  {
LABEL_3:
    v4 = sub_1649960(v3);
    if ( v5 > 0xA && *(_QWORD *)v4 == 0x675F6D766C6C5F5FLL && *((_WORD *)v4 + 4) == 28515 && v4[10] == 118 )
      return 0;
    v6 = sub_1649960(v3);
    if ( v7 > 0xA && *(_QWORD *)v6 == 0x675F6D766C6C5F5FLL && *((_WORD *)v6 + 4) == 25699 && v6[10] == 97 )
      return 0;
    goto LABEL_8;
  }
  v16 = a1 + 240;
  v10 = sub_15E61A0(v2);
  v12 = v11;
  v17 = 260;
  sub_16E1010((__int64)&s2, (__int64)&v16);
  v13 = v21;
  if ( s2 != v20 )
  {
    v15 = v21;
    j_j___libc_free_0(s2, v20[0] + 1LL);
    v13 = v15;
  }
  sub_1694890((__int64)&s2, 1, v13, 0);
  v14 = s2;
  if ( v12 < n || n && memcmp((const void *)(v10 + v12 - n), s2, n) )
  {
    if ( v14 != v20 )
      j_j___libc_free_0(v14, v20[0] + 1LL);
    goto LABEL_3;
  }
  if ( v14 == v20 )
    return 0;
  j_j___libc_free_0(v14, v20[0] + 1LL);
  return 0;
}
