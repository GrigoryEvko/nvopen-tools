// Function: sub_C81CA0
// Address: 0xc81ca0
//
__int64 __fastcall sub_C81CA0(__int64 a1, unsigned int a2)
{
  bool v3; // zf
  unsigned __int8 v4; // al
  char **v5; // rax
  char *v6; // rdi
  size_t v7; // rsi
  __int64 v8; // rdx
  char *v10; // [rsp+10h] [rbp-B0h] BYREF
  size_t v11; // [rsp+18h] [rbp-A8h]
  __int64 v12; // [rsp+20h] [rbp-A0h]
  _BYTE v13[152]; // [rsp+28h] [rbp-98h] BYREF

  v3 = *(_BYTE *)(a1 + 33) == 1;
  v10 = v13;
  v11 = 0;
  v12 = 128;
  if ( !v3 )
    goto LABEL_6;
  v4 = *(_BYTE *)(a1 + 32);
  if ( v4 == 1 )
  {
    v7 = 0;
    v6 = 0;
    goto LABEL_7;
  }
  if ( (unsigned __int8)(v4 - 3) > 3u )
  {
LABEL_6:
    sub_CA0EC0(a1, &v10);
    v7 = v11;
    v6 = v10;
    goto LABEL_7;
  }
  if ( v4 == 4 )
  {
    v5 = *(char ***)a1;
    v6 = **(char ***)a1;
    v7 = (size_t)v5[1];
    goto LABEL_7;
  }
  if ( v4 > 4u )
  {
    if ( (unsigned __int8)(v4 - 5) <= 1u )
    {
      v7 = *(_QWORD *)(a1 + 8);
      v6 = *(char **)a1;
      goto LABEL_7;
    }
LABEL_17:
    BUG();
  }
  if ( v4 != 3 )
    goto LABEL_17;
  v6 = *(char **)a1;
  v7 = 0;
  if ( v6 )
    v7 = strlen(v6);
LABEL_7:
  sub_C80DA0(v6, v7, a2);
  LOBYTE(a2) = v8 != 0;
  if ( v10 != v13 )
    _libc_free(v10, v7);
  return a2;
}
