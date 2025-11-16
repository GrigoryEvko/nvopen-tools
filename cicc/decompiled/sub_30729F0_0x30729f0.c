// Function: sub_30729F0
// Address: 0x30729f0
//
unsigned __int64 __fastcall sub_30729F0(__int64 a1, _BYTE **a2, int a3, __int64 *a4, __int64 a5, __int64 a6)
{
  char v6; // r11
  unsigned __int64 v7; // r12
  __int64 v9; // r13
  _BYTE **v10; // r14
  unsigned __int8 v11; // al
  __int64 v12; // r15
  _BYTE *v13; // rsi
  int v14; // edi
  __int64 v15; // rcx
  __int64 v16; // rdx
  char *v17; // rax
  char v18; // dl
  signed __int64 v19; // rax
  int v20; // edx
  bool v21; // zf
  int v22; // edx
  bool v23; // of
  int v26; // [rsp+1Ch] [rbp-74h]
  __int64 v27; // [rsp+20h] [rbp-70h] BYREF
  char *v28; // [rsp+28h] [rbp-68h]
  __int64 v29; // [rsp+30h] [rbp-60h]
  int v30; // [rsp+38h] [rbp-58h]
  char v31; // [rsp+3Ch] [rbp-54h]
  char v32; // [rsp+40h] [rbp-50h] BYREF

  v27 = 0;
  v28 = &v32;
  v29 = 4;
  v30 = 0;
  v31 = 1;
  if ( !a3 )
    return 0;
  v6 = 1;
  v7 = 0;
  v26 = 0;
  v9 = (__int64)&a2[(unsigned int)(a3 - 1) + 1];
  v10 = a2;
  do
  {
    while ( 1 )
    {
      v12 = *a4;
      v13 = *v10;
      v14 = *(unsigned __int8 *)(*a4 + 8);
      v15 = (unsigned int)(v14 - 17);
      v16 = *(unsigned __int8 *)(*a4 + 8);
      if ( (unsigned int)v15 > 1 )
      {
        if ( (_BYTE)v14 == 12 )
          goto LABEL_8;
LABEL_12:
        v11 = *(_BYTE *)(*a4 + 8);
        if ( v14 == 17 )
          v11 = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
        goto LABEL_5;
      }
      v11 = *(_BYTE *)(**(_QWORD **)(v12 + 16) + 8LL);
      if ( v11 == 12 )
        goto LABEL_8;
      if ( v14 != 18 )
        goto LABEL_12;
LABEL_5:
      if ( v11 > 3u && v11 != 5 && (v11 & 0xFD) != 4 )
        break;
LABEL_8:
      if ( *v13 > 0x15u )
        goto LABEL_18;
LABEL_9:
      ++v10;
      ++a4;
      if ( v10 == (_BYTE **)v9 )
        goto LABEL_29;
    }
    if ( (unsigned int)v15 <= 1 )
      v16 = *(unsigned __int8 *)(**(_QWORD **)(v12 + 16) + 8LL);
    if ( (_BYTE)v16 != 14 || *v13 <= 0x15u )
      goto LABEL_9;
LABEL_18:
    if ( !v6 )
      goto LABEL_23;
    v17 = v28;
    v15 = HIDWORD(v29);
    v16 = (__int64)&v28[8 * HIDWORD(v29)];
    if ( v28 != (char *)v16 )
    {
      while ( v13 != *(_BYTE **)v17 )
      {
        v17 += 8;
        if ( (char *)v16 == v17 )
          goto LABEL_22;
      }
      goto LABEL_9;
    }
LABEL_22:
    if ( HIDWORD(v29) < (unsigned int)v29 )
    {
      ++HIDWORD(v29);
      *(_QWORD *)v16 = v13;
      v6 = v31;
      ++v27;
    }
    else
    {
LABEL_23:
      sub_C8CC70((__int64)&v27, (__int64)v13, v16, v15, a5, a6);
      v6 = v31;
      if ( !v18 )
        goto LABEL_9;
    }
    if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 > 1 )
      goto LABEL_9;
    v19 = sub_30727B0(a1, v12, 0, 1);
    v21 = v20 == 1;
    v22 = 1;
    if ( !v21 )
      v22 = v26;
    v23 = __OFADD__(v19, v7);
    v7 += v19;
    v26 = v22;
    if ( v23 )
    {
      v7 = 0x8000000000000000LL;
      if ( v19 > 0 )
        v7 = 0x7FFFFFFFFFFFFFFFLL;
    }
    ++v10;
    v6 = v31;
    ++a4;
  }
  while ( v10 != (_BYTE **)v9 );
LABEL_29:
  if ( !v6 )
    _libc_free((unsigned __int64)v28);
  return v7;
}
