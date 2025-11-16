// Function: sub_C94C70
// Address: 0xc94c70
//
char *__fastcall sub_C94C70(__int64 a1, __int64 a2)
{
  bool v2; // zf
  unsigned __int8 v3; // al
  char **v4; // rax
  char *v5; // rsi
  size_t v6; // rdx
  char *v7; // r12
  char *v9; // [rsp+10h] [rbp-C0h] BYREF
  size_t v10; // [rsp+18h] [rbp-B8h]
  __int64 v11; // [rsp+20h] [rbp-B0h]
  _BYTE v12[168]; // [rsp+28h] [rbp-A8h] BYREF

  v2 = *(_BYTE *)(a2 + 33) == 1;
  v9 = v12;
  v10 = 0;
  v11 = 128;
  if ( !v2 )
    goto LABEL_6;
  v3 = *(_BYTE *)(a2 + 32);
  if ( v3 == 1 )
  {
    v6 = 0;
    v5 = 0;
    goto LABEL_7;
  }
  if ( (unsigned __int8)(v3 - 3) > 3u )
  {
LABEL_6:
    sub_CA0EC0(a2, &v9);
    v6 = v10;
    v5 = v9;
    goto LABEL_7;
  }
  if ( v3 == 4 )
  {
    v4 = *(char ***)a2;
    v5 = **(char ***)a2;
    v6 = (size_t)v4[1];
    goto LABEL_7;
  }
  if ( v3 > 4u )
  {
    if ( (unsigned __int8)(v3 - 5) <= 1u )
    {
      v6 = *(_QWORD *)(a2 + 8);
      v5 = *(char **)a2;
      goto LABEL_7;
    }
LABEL_17:
    BUG();
  }
  if ( v3 != 3 )
    goto LABEL_17;
  v5 = *(char **)a2;
  v6 = 0;
  if ( v5 )
    v6 = strlen(v5);
LABEL_7:
  v7 = sub_C94910(a1, v5, v6);
  if ( v9 != v12 )
    _libc_free(v9, v5);
  return v7;
}
