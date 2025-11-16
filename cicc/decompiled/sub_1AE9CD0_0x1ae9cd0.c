// Function: sub_1AE9CD0
// Address: 0x1ae9cd0
//
__int64 __fastcall sub_1AE9CD0(__int64 a1)
{
  __int64 v1; // rsi
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 i; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  _BYTE *v9; // rdi
  unsigned int v10; // r12d
  __int64 *v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // r13
  __int64 v15; // r15
  _QWORD *v16; // rax
  unsigned __int8 v17; // cl
  int v19; // edx
  __int64 v20; // rax
  _QWORD *v21; // r15
  unsigned __int8 v22; // al
  __int64 v23; // [rsp+8h] [rbp-268h]
  __int64 v24; // [rsp+20h] [rbp-250h]
  __int64 *v25; // [rsp+28h] [rbp-248h]
  __int64 v26; // [rsp+38h] [rbp-238h] BYREF
  _BYTE *v27; // [rsp+40h] [rbp-230h] BYREF
  __int64 v28; // [rsp+48h] [rbp-228h]
  _BYTE v29[32]; // [rsp+50h] [rbp-220h] BYREF
  _QWORD v30[64]; // [rsp+70h] [rbp-200h] BYREF

  v1 = *(_QWORD *)(a1 + 40);
  sub_15A5590((__int64)v30, (__int64 *)v1, 0, 0);
  v4 = *(_QWORD *)(a1 + 80);
  v27 = v29;
  v28 = 0x400000000LL;
  if ( v4 == a1 + 72 )
  {
    v10 = 0;
    goto LABEL_33;
  }
  do
  {
    if ( !v4 )
      BUG();
    v5 = *(_QWORD *)(v4 + 24);
    for ( i = v4 + 16; i != v5; v5 = *(_QWORD *)(v5 + 8) )
    {
      while ( 1 )
      {
        if ( !v5 )
          BUG();
        if ( *(_BYTE *)(v5 - 8) == 78 )
        {
          v7 = *(_QWORD *)(v5 - 48);
          if ( !*(_BYTE *)(v7 + 16) && (*(_BYTE *)(v7 + 33) & 0x20) != 0 && *(_DWORD *)(v7 + 36) == 36 )
            break;
        }
        v5 = *(_QWORD *)(v5 + 8);
        if ( i == v5 )
          goto LABEL_14;
      }
      v8 = (unsigned int)v28;
      if ( (unsigned int)v28 >= HIDWORD(v28) )
      {
        v1 = (__int64)v29;
        sub_16CD150((__int64)&v27, v29, 0, 8, v2, v3);
        v8 = (unsigned int)v28;
      }
      *(_QWORD *)&v27[8 * v8] = v5 - 24;
      LODWORD(v28) = v28 + 1;
    }
LABEL_14:
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( a1 + 72 != v4 );
  v9 = v27;
  v10 = 0;
  if ( !(_DWORD)v28 )
    goto LABEL_31;
  v11 = (__int64 *)v27;
  v25 = (__int64 *)&v27[8 * (unsigned int)v28];
  while ( 1 )
  {
LABEL_18:
    v12 = *v11;
    v1 = 1;
    v13 = sub_1601A30(*v11, 1);
    v14 = (_QWORD *)v13;
    if ( !v13
      || *(_BYTE *)(v13 + 16) != 53
      || (unsigned __int8)sub_15F8BF0(v13)
      || *(_BYTE *)(*(_QWORD *)(*v14 + 24LL) + 8LL) == 14 )
    {
      goto LABEL_17;
    }
    v24 = v14[1];
    if ( v24 )
      break;
LABEL_43:
    sub_15F20C0((_QWORD *)v12);
LABEL_17:
    if ( v25 == ++v11 )
      goto LABEL_30;
  }
  v15 = v14[1];
  while ( 1 )
  {
    v16 = sub_1648700(v15);
    v17 = *((_BYTE *)v16 + 16);
    if ( v17 > 0x17u && (v17 == 54 || v17 == 55) && (*((_BYTE *)v16 + 18) & 1) != 0 )
      break;
    v15 = *(_QWORD *)(v15 + 8);
    if ( !v15 )
    {
      do
      {
        v21 = sub_1648700(v24);
        v22 = *((_BYTE *)v21 + 16);
        if ( v22 > 0x17u )
        {
          switch ( v22 )
          {
            case '7':
              if ( (unsigned int)sub_1648720(v24) == 1 )
              {
                v1 = (__int64)v21;
                sub_1AE9B50(v12, (__int64)v21, v30);
              }
              break;
            case '6':
              v1 = (__int64)v21;
              sub_1AE9C10(v12, v21, v30);
              break;
            case 'N':
              v19 = *(_DWORD *)(v12 + 20);
              v26 = 6;
              v23 = sub_15C49B0(*(_QWORD **)(*(_QWORD *)(v12 + 24 * (2LL - (v19 & 0xFFFFFFF))) + 24LL), &v26, 1);
              v20 = sub_15C70A0(v12 + 48);
              v1 = (__int64)v14;
              sub_15A76D0(
                v30,
                (__int64)v14,
                *(_QWORD *)(*(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF))) + 24LL),
                v23,
                v20,
                (__int64)v21);
              break;
          }
        }
        v24 = *(_QWORD *)(v24 + 8);
      }
      while ( v24 );
      goto LABEL_43;
    }
  }
  if ( v25 != ++v11 )
    goto LABEL_18;
LABEL_30:
  v9 = v27;
  v10 = 1;
LABEL_31:
  if ( v9 != v29 )
    _libc_free((unsigned __int64)v9);
LABEL_33:
  sub_129E320((__int64)v30, v1);
  return v10;
}
