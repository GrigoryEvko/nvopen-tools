// Function: sub_BF9C00
// Address: 0xbf9c00
//
void __fastcall sub_BF9C00(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  int v4; // edx
  __int16 v5; // dx
  unsigned __int64 v6; // rcx
  const char *v7; // rax
  const char *v8; // rax
  __int64 v9; // r14
  _BYTE *v10; // rax
  __int64 v11; // rax
  _QWORD v12[4]; // [rsp+0h] [rbp-90h] BYREF
  char v13; // [rsp+20h] [rbp-70h]
  char v14; // [rsp+21h] [rbp-6Fh]
  __int64 v15; // [rsp+30h] [rbp-60h] BYREF
  char *v16; // [rsp+38h] [rbp-58h]
  __int64 v17; // [rsp+40h] [rbp-50h]
  int v18; // [rsp+48h] [rbp-48h]
  char v19; // [rsp+4Ch] [rbp-44h]
  char v20; // [rsp+50h] [rbp-40h] BYREF

  v2 = a2;
  v3 = *(_QWORD *)(a2 + 72);
  v16 = &v20;
  v17 = 4;
  v18 = 0;
  v19 = 1;
  v4 = *(unsigned __int8 *)(v3 + 8);
  v15 = 0;
  if ( (_BYTE)v4 != 12 && (unsigned __int8)v4 > 3u && (_BYTE)v4 != 5 && (v4 & 0xFD) != 4 && (v4 & 0xFB) != 0xA )
  {
    if ( (unsigned __int8)(v4 - 15) > 3u && v4 != 20
      || (a2 = (__int64)&v15, !(unsigned __int8)sub_BCEBA0(v3, (__int64)&v15)) )
    {
      v14 = 1;
      v8 = "Cannot allocate unsized type";
      goto LABEL_18;
    }
  }
  if ( (unsigned __int8)sub_BCF0D0(v3) )
  {
    v14 = 1;
    v8 = "Alloca has illegal target extension type";
    goto LABEL_18;
  }
  if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v2 - 32) + 8LL) + 8LL) != 12 )
  {
    v14 = 1;
    v8 = "Alloca array size must have integer type";
    goto LABEL_18;
  }
  v5 = *(_WORD *)(v2 + 2);
  _BitScanReverse64(&v6, 1LL << v5);
  if ( 0x8000000000000000LL >> ((unsigned __int8)v6 ^ 0x3Fu) > 0x100000000LL )
  {
    v14 = 1;
    v7 = "huge alignment values are unsupported";
LABEL_9:
    a2 = (__int64)v12;
    v12[0] = v7;
    v13 = 3;
    sub_BDBF70((__int64 *)a1, (__int64)v12);
    if ( *(_QWORD *)a1 )
    {
LABEL_22:
      a2 = v2;
      sub_BDBD80(a1, (_BYTE *)v2);
    }
LABEL_23:
    if ( v19 )
      return;
LABEL_28:
    _libc_free(v16, a2);
    return;
  }
  if ( (v5 & 0x80u) != 0 )
  {
    if ( *(_BYTE *)(v3 + 8) != 14 )
    {
      v14 = 1;
      v7 = "swifterror alloca must have pointer type";
      goto LABEL_9;
    }
    if ( !(unsigned __int8)sub_B4CE70(v2) )
    {
      sub_BDC190(a1, v2);
      goto LABEL_27;
    }
    v14 = 1;
    v8 = "swifterror alloca must not be array allocation";
LABEL_18:
    v9 = *(_QWORD *)a1;
    v12[0] = v8;
    v13 = 3;
    if ( v9 )
    {
      a2 = v9;
      sub_CA0E80(v12, v9);
      v10 = *(_BYTE **)(v9 + 32);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
      {
        a2 = 10;
        sub_CB5D20(v9, 10);
      }
      else
      {
        *(_QWORD *)(v9 + 32) = v10 + 1;
        *v10 = 10;
      }
      v11 = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 152) = 1;
      if ( v11 )
        goto LABEL_22;
    }
    else
    {
      *(_BYTE *)(a1 + 152) = 1;
    }
    goto LABEL_23;
  }
LABEL_27:
  a2 = v2;
  sub_BF6FE0(a1, v2);
  if ( !v19 )
    goto LABEL_28;
}
