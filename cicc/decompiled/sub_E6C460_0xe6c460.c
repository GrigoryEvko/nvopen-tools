// Function: sub_E6C460
// Address: 0xe6c460
//
__int64 __fastcall sub_E6C460(__int64 a1, const char **a2)
{
  bool v2; // zf
  unsigned __int8 v3; // al
  const char *v4; // rcx
  size_t v5; // r15
  _QWORD *v6; // rsi
  unsigned __int64 v7; // rax
  const char *v8; // rcx
  __int64 v9; // r13
  __int64 v10; // rbx
  __int64 v12; // rax
  size_t v13; // rdx
  unsigned __int8 v14; // dl
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rax
  size_t v18; // rax
  const char *s1; // [rsp+8h] [rbp-108h]
  const char *s1a; // [rsp+8h] [rbp-108h]
  _QWORD v21[4]; // [rsp+10h] [rbp-100h] BYREF
  __int16 v22; // [rsp+30h] [rbp-E0h]
  const char *v23; // [rsp+40h] [rbp-D0h] BYREF
  size_t v24; // [rsp+48h] [rbp-C8h]
  __int64 v25; // [rsp+50h] [rbp-C0h]
  _BYTE v26[184]; // [rsp+58h] [rbp-B8h] BYREF

  v2 = *((_BYTE *)a2 + 33) == 1;
  v23 = v26;
  v24 = 0;
  v25 = 128;
  if ( !v2 )
    goto LABEL_6;
  v3 = *((_BYTE *)a2 + 32);
  if ( v3 == 1 )
  {
    v5 = 0;
    v4 = 0;
    goto LABEL_7;
  }
  if ( (unsigned __int8)(v3 - 3) > 3u )
  {
LABEL_6:
    sub_CA0EC0((__int64)a2, (__int64)&v23);
    v5 = v24;
    v4 = v23;
    goto LABEL_7;
  }
  if ( v3 == 4 )
  {
    v4 = *(const char **)*a2;
    v5 = *((_QWORD *)*a2 + 1);
    goto LABEL_7;
  }
  if ( v3 > 4u )
  {
    if ( (unsigned __int8)(v3 - 5) <= 1u )
    {
      v5 = (size_t)a2[1];
      v4 = *a2;
      goto LABEL_7;
    }
LABEL_25:
    BUG();
  }
  if ( v3 != 3 )
    goto LABEL_25;
  v4 = *a2;
  v5 = 0;
  if ( *a2 )
  {
    s1a = *a2;
    v18 = strlen(*a2);
    v4 = s1a;
    v5 = v18;
  }
LABEL_7:
  v6 = v4;
  s1 = v4;
  v7 = sub_E6B3F0(a1, v4, v5);
  v8 = s1;
  v9 = *(_QWORD *)(v7 + 8);
  v10 = v7;
  if ( v9 )
    goto LABEL_8;
  v12 = *(_QWORD *)(a1 + 152);
  v13 = *(_QWORD *)(v12 + 96);
  if ( v13 <= v5 )
  {
    if ( !v13 || (v16 = memcmp(s1, *(const void **)(v12 + 88), v13), v8 = s1, !v16) )
    {
      v14 = *(_BYTE *)(a1 + 1907) ^ 1;
      if ( *(_BYTE *)(v10 + 20) )
        goto LABEL_17;
LABEL_13:
      *(_BYTE *)(v10 + 20) = 1;
      v6 = (_QWORD *)v10;
      v15 = sub_E6BCB0((_DWORD *)a1, v10, v14);
      *(_QWORD *)(v10 + 8) = v15;
      v9 = v15;
      goto LABEL_8;
    }
  }
  v14 = 0;
  if ( !*(_BYTE *)(v10 + 20) )
    goto LABEL_13;
LABEL_17:
  v21[0] = v8;
  v6 = v21;
  v22 = 261;
  v21[1] = v5;
  v17 = sub_E6BFC0((_DWORD *)a1, (__int64)v21, 0, v14);
  *(_QWORD *)(v10 + 8) = v17;
  v9 = v17;
LABEL_8:
  if ( v23 != v26 )
    _libc_free(v23, v6);
  return v9;
}
