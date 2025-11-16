// Function: sub_ECA880
// Address: 0xeca880
//
__int64 __fastcall sub_ECA880(__int64 a1)
{
  __int64 v2; // rdi
  char v3; // r12
  unsigned int v4; // r12d
  _BYTE *v5; // r14
  _BYTE *v6; // rax
  __int64 v7; // r13
  const char *v8; // rax
  __int64 v9; // rdi
  _BOOL4 v11; // r15d
  __int64 v12; // r13
  void (__fastcall *v13)(__int64, __int64, void *, size_t, _BOOL4); // r14
  __int64 v14; // rax
  __int64 v15; // rax
  const char *v16; // [rsp+0h] [rbp-90h] BYREF
  const char *v17; // [rsp+8h] [rbp-88h]
  void *s; // [rsp+10h] [rbp-80h] BYREF
  size_t n; // [rsp+18h] [rbp-78h]
  __int64 v20; // [rsp+20h] [rbp-70h] BYREF
  __int64 v21; // [rsp+28h] [rbp-68h]
  const char *v22[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v23; // [rsp+50h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 8);
  v16 = 0;
  v17 = 0;
  s = 0;
  n = 0;
  v20 = 0;
  v21 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v16) )
    goto LABEL_12;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 26 )
  {
    HIBYTE(v23) = 1;
    v8 = "expected a comma";
    goto LABEL_13;
  }
  v3 = *(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 113);
  *(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 113) = 1;
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  *(_BYTE *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 113) = v3;
  v4 = (*(__int64 (__fastcall **)(_QWORD, void **))(**(_QWORD **)(a1 + 8) + 192LL))(*(_QWORD *)(a1 + 8), &s);
  if ( (_BYTE)v4 )
  {
LABEL_12:
    HIBYTE(v23) = 1;
    v8 = "expected identifier";
  }
  else if ( n && (v5 = s, (v6 = memchr(s, 64, n)) != 0) && v6 - v5 != -1 )
  {
    v7 = sub_C931B0((__int64 *)&s, word_3F645A0, 3u, 0);
    if ( !(unsigned __int8)sub_ECE2A0(*(_QWORD *)(a1 + 8), 26) )
    {
      v11 = v7 == -1;
LABEL_18:
      sub_ECE2A0(*(_QWORD *)(a1 + 8), 9);
      v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v13 = *(void (__fastcall **)(__int64, __int64, void *, size_t, _BOOL4))(*(_QWORD *)v12 + 456LL);
      v14 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v23 = 261;
      v22[0] = v16;
      v22[1] = v17;
      v15 = sub_E6C460(v14, v22);
      v13(v12, v15, s, n, v11);
      return v4;
    }
    if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 192LL))(
            *(_QWORD *)(a1 + 8),
            &v20)
      && v21 == 6
      && *(_DWORD *)v20 == 1869440370 )
    {
      v11 = 0;
      if ( *(_WORD *)(v20 + 4) == 25974 )
        goto LABEL_18;
    }
    HIBYTE(v23) = 1;
    v8 = "expected 'remove'";
  }
  else
  {
    HIBYTE(v23) = 1;
    v8 = "expected a '@' in the name";
  }
LABEL_13:
  v9 = *(_QWORD *)(a1 + 8);
  v22[0] = v8;
  LOBYTE(v23) = 3;
  return (unsigned int)sub_ECE0E0(v9, v22, 0, 0);
}
