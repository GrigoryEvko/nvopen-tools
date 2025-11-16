// Function: sub_E99A90
// Address: 0xe99a90
//
__int64 __fastcall sub_E99A90(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  bool v3; // zf
  void (__noreturn *v4)(); // r13
  unsigned __int8 v5; // al
  const char **v6; // rax
  const char *v7; // rsi
  size_t v8; // rdx
  __int64 result; // rax
  const char *v10; // [rsp+10h] [rbp-C0h] BYREF
  size_t v11; // [rsp+18h] [rbp-B8h]
  __int64 v12; // [rsp+20h] [rbp-B0h]
  _BYTE v13[168]; // [rsp+28h] [rbp-A8h] BYREF

  v2 = *a1;
  v3 = *(_BYTE *)(a2 + 33) == 1;
  v10 = v13;
  v11 = 0;
  v12 = 128;
  v4 = *(void (__noreturn **)())(v2 + 40);
  if ( !v3 )
    goto LABEL_6;
  v5 = *(_BYTE *)(a2 + 32);
  if ( v5 == 1 )
  {
    v8 = 0;
    v7 = 0;
    goto LABEL_7;
  }
  if ( (unsigned __int8)(v5 - 3) > 3u )
  {
LABEL_6:
    sub_CA0EC0(a2, (__int64)&v10);
    v8 = v11;
    v7 = v10;
    goto LABEL_7;
  }
  if ( v5 == 4 )
  {
    v6 = *(const char ***)a2;
    v7 = **(const char ***)a2;
    v8 = (size_t)v6[1];
    goto LABEL_7;
  }
  if ( v5 > 4u )
  {
    if ( (unsigned __int8)(v5 - 5) <= 1u )
    {
      v8 = *(_QWORD *)(a2 + 8);
      v7 = *(const char **)a2;
      goto LABEL_7;
    }
LABEL_19:
    BUG();
  }
  if ( v5 != 3 )
    goto LABEL_19;
  v7 = *(const char **)a2;
  v8 = 0;
  if ( v7 )
    v8 = strlen(v7);
LABEL_7:
  if ( v4 == sub_E979C0 )
    sub_C64ED0(
      "EmitRawText called on an MCStreamer that doesn't support it (target backend is likely missing an AsmStreamer implementation)",
      1u);
  result = ((__int64 (__fastcall *)(__int64 *, const char *, size_t))v4)(a1, v7, v8);
  if ( v10 != v13 )
    return _libc_free(v10, v7);
  return result;
}
