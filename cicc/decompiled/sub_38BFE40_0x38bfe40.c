// Function: sub_38BFE40
// Address: 0x38bfe40
//
__int64 __fastcall sub_38BFE40(__int64 a1)
{
  _QWORD *v2; // rax
  char *v3; // rsi
  size_t v4; // r14
  char *v5; // rdi
  unsigned __int64 v6; // rax
  _QWORD *v7; // r15
  int v8; // r9d
  __int64 v9; // r12
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v14; // [rsp+10h] [rbp-E0h]
  void *dest; // [rsp+18h] [rbp-D8h]
  int v16; // [rsp+20h] [rbp-D0h]
  void **v17; // [rsp+28h] [rbp-C8h]
  _BYTE *v18; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v19; // [rsp+38h] [rbp-B8h]
  _BYTE v20[176]; // [rsp+40h] [rbp-B0h] BYREF

  v19 = 0x8000000000LL;
  v17 = (void **)&v18;
  v18 = v20;
  v13[0] = &unk_49EFC48;
  v16 = 1;
  dest = 0;
  v14 = 0;
  v13[1] = 0;
  sub_16E7A40((__int64)v13, 0, 0, 0);
  v2 = *(_QWORD **)(a1 + 16);
  v3 = (char *)v2[14];
  if ( *v3 )
  {
    v4 = v2[15];
  }
  else
  {
    v3 = (char *)v2[10];
    v4 = v2[11];
  }
  v5 = (char *)dest;
  v6 = v14 - (_QWORD)dest;
  if ( v14 - (__int64)dest < v4 )
  {
    v12 = sub_16E7EE0((__int64)v13, v3, v4);
    v5 = *(char **)(v12 + 24);
    v7 = (_QWORD *)v12;
    v6 = *(_QWORD *)(v12 + 16) - (_QWORD)v5;
  }
  else
  {
    v7 = v13;
    if ( v4 )
    {
      memcpy(dest, v3, v4);
      v5 = (char *)dest + v4;
      v11 = v14 - ((_QWORD)dest + v4);
      dest = (char *)dest + v4;
      if ( v11 > 2 )
        goto LABEL_6;
      goto LABEL_11;
    }
  }
  if ( v6 > 2 )
  {
LABEL_6:
    v5[2] = 112;
    *(_WORD *)v5 = 28020;
    v7[3] += 3LL;
    goto LABEL_7;
  }
LABEL_11:
  sub_16E7EE0((__int64)v7, "tmp", 3u);
LABEL_7:
  v13[0] = &unk_49EFD28;
  sub_16E7960((__int64)v13);
  v9 = sub_38BEE30(a1, v18, (unsigned int)v19, 1, 0, v8);
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
  return v9;
}
