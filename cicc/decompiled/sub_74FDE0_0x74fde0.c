// Function: sub_74FDE0
// Address: 0x74fde0
//
__int64 __fastcall sub_74FDE0(__int64 a1, int a2, __int64 a3)
{
  bool v5; // zf
  __int64 *v6; // r14
  char v7; // al
  char v8; // al
  char v9; // al
  __int64 v10; // rdi
  char **v11; // rax
  char v12; // di
  _DWORD v14[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(_BYTE *)(a3 + 136) == 0;
  v14[0] = a2;
  if ( !v5 && !dword_4F068C4 )
    return v14[0];
  v6 = *(__int64 **)(a1 + 104);
  if ( v6 )
  {
    sub_74A140(27, *(__int64 **)(a1 + 104), v14, (void (__fastcall **)(char *))a3);
    sub_74A140(29, v6, v14, (void (__fastcall **)(char *))a3);
    sub_74A140(32, v6, v14, (void (__fastcall **)(char *))a3);
    sub_74A140(37, v6, v14, (void (__fastcall **)(char *))a3);
    sub_74A140(39, v6, v14, (void (__fastcall **)(char *))a3);
    sub_74A140(41, v6, v14, (void (__fastcall **)(char *))a3);
    sub_74A140(45, v6, v14, (void (__fastcall **)(char *))a3);
    sub_74A140(77, v6, v14, (void (__fastcall **)(char *))a3);
  }
  v7 = *(_BYTE *)(a1 + 200);
  if ( (v7 & 8) == 0 )
  {
LABEL_8:
    if ( (v7 & 0x10) == 0 )
      goto LABEL_9;
    goto LABEL_37;
  }
  if ( (*(_BYTE *)(a1 + 207) & 4) != 0 )
  {
    sub_745D60(
      "__constructor__",
      *(unsigned __int16 *)(*(_QWORD *)(a1 + 256) + 32LL),
      v14,
      (__int64 (__fastcall **)(const char *, _QWORD))a3);
    v7 = *(_BYTE *)(a1 + 200);
    goto LABEL_8;
  }
  sub_7450F0("__constructor__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 200) & 0x10) == 0 )
    goto LABEL_9;
LABEL_37:
  if ( (*(_BYTE *)(a1 + 207) & 8) == 0 )
  {
    sub_7450F0("__destructor__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
LABEL_9:
    if ( *(char *)(a1 + 196) >= 0 )
      goto LABEL_10;
    goto LABEL_39;
  }
  sub_745D60(
    "__destructor__",
    *(unsigned __int16 *)(*(_QWORD *)(a1 + 256) + 34LL),
    v14,
    (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( *(char *)(a1 + 196) >= 0 )
  {
LABEL_10:
    if ( (*(_BYTE *)(a1 + 200) & 0x60) != 0x20 )
      goto LABEL_11;
    goto LABEL_40;
  }
LABEL_39:
  sub_7450F0("__pure__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 200) & 0x60) != 0x20 )
  {
LABEL_11:
    if ( (*(_BYTE *)(a1 + 91) & 4) == 0 )
      goto LABEL_12;
    goto LABEL_41;
  }
LABEL_40:
  sub_7450F0("__weak__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 91) & 4) == 0 )
  {
LABEL_12:
    if ( (*(_BYTE *)(a1 + 201) & 2) == 0 )
      goto LABEL_13;
    goto LABEL_42;
  }
LABEL_41:
  sub_7450F0("__unused__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 201) & 2) == 0 )
  {
LABEL_13:
    if ( *(char *)(a1 + 90) >= 0 )
      goto LABEL_14;
    goto LABEL_43;
  }
LABEL_42:
  sub_7450F0("__used__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( *(char *)(a1 + 90) >= 0 )
    goto LABEL_14;
LABEL_43:
  if ( *(_BYTE *)(a3 + 136) && *(_BYTE *)(a3 + 141) )
    goto LABEL_14;
  if ( sub_736C60(6, *(__int64 **)(a1 + 104)) )
    sub_7450F0("__deprecated__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( !sub_736C60(21, *(__int64 **)(a1 + 104)) )
  {
LABEL_14:
    v8 = *(_BYTE *)(a1 + 201);
    if ( (v8 & 0x20) == 0 )
      goto LABEL_15;
    goto LABEL_49;
  }
  sub_7450F0("__unavailable__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  v8 = *(_BYTE *)(a1 + 201);
  if ( (v8 & 0x20) == 0 )
  {
LABEL_15:
    if ( (v8 & 0x40) == 0 )
      goto LABEL_16;
    goto LABEL_50;
  }
LABEL_49:
  sub_7450F0("__malloc__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  v8 = *(_BYTE *)(a1 + 201);
  if ( (v8 & 0x40) == 0 )
  {
LABEL_16:
    if ( v8 >= 0 )
      goto LABEL_17;
    goto LABEL_51;
  }
LABEL_50:
  sub_7450F0("__no_instrument_function__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( *(char *)(a1 + 201) >= 0 )
  {
LABEL_17:
    if ( (*(_BYTE *)(a1 + 196) & 0x40) == 0 )
      goto LABEL_18;
LABEL_52:
    sub_7450F0("__noinline__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
    v9 = *(_BYTE *)(a1 + 202);
    if ( (v9 & 1) == 0 )
      goto LABEL_19;
    goto LABEL_53;
  }
LABEL_51:
  sub_7450F0("__no_check_memory_usage__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  if ( (*(_BYTE *)(a1 + 196) & 0x40) != 0 )
    goto LABEL_52;
LABEL_18:
  v9 = *(_BYTE *)(a1 + 202);
  if ( (v9 & 1) == 0 )
    goto LABEL_19;
LABEL_53:
  sub_7450F0("__always_inline__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  v9 = *(_BYTE *)(a1 + 202);
LABEL_19:
  if ( (v9 & 2) != 0 )
  {
    if ( !dword_4F068D0 )
    {
      if ( !unk_4F068E0 )
        goto LABEL_22;
      if ( qword_4F068D8 <= 0x9D07u )
      {
        if ( (*(_BYTE *)(a1 + 195) & 0x10) == 0 )
          goto LABEL_22;
        goto LABEL_34;
      }
    }
    sub_7450F0("__gnu_inline__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  }
  if ( (*(_BYTE *)(a1 + 195) & 0x10) != 0 )
  {
    if ( dword_4F068D0 )
      goto LABEL_35;
    if ( unk_4F068E0 )
    {
LABEL_34:
      if ( qword_4F068D8 > 0x765Bu )
      {
LABEL_35:
        sub_7450F0("__nothrow__", v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
        v10 = *(_QWORD *)(a1 + 152);
        if ( *(_BYTE *)(v10 + 140) != 7 )
          goto LABEL_23;
        goto LABEL_36;
      }
    }
  }
LABEL_22:
  v10 = *(_QWORD *)(a1 + 152);
  if ( *(_BYTE *)(v10 + 140) != 7 )
    goto LABEL_23;
LABEL_36:
  sub_745EC0(v10, v14, a3);
LABEL_23:
  v11 = *(char ***)(a1 + 256);
  if ( v11 && *v11 )
    sub_747170(*v11, v14, a3);
  v12 = *(_BYTE *)(a1 + 200);
  if ( v12 < 0 )
  {
    sub_74A140(26, *(__int64 **)(a1 + 104), v14, (void (__fastcall **)(char *))a3);
    sub_745150(*(_BYTE *)(a1 + 200) & 7, v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  }
  else if ( (v12 & 0x40) != 0 )
  {
    sub_74A140(79, *(__int64 **)(a1 + 104), v14, (void (__fastcall **)(char *))a3);
    sub_745150(*(_BYTE *)(a1 + 200) & 7, v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  }
  else
  {
    sub_745150(v12 & 7, v14, (__int64 (__fastcall **)(const char *, _QWORD))a3);
  }
  return v14[0];
}
