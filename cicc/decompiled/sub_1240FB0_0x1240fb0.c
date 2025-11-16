// Function: sub_1240FB0
// Address: 0x1240fb0
//
__int64 __fastcall sub_1240FB0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r13
  __int64 v5; // rsi
  int v6; // eax
  __int64 v7; // r9
  __int64 v8; // rdx
  unsigned __int64 v9; // r8
  __int64 v10; // rdx
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // rsi
  char v14; // dl
  __int64 v15; // rdx
  char v16; // [rsp+8h] [rbp-138h]
  int v17; // [rsp+18h] [rbp-128h]
  char v18; // [rsp+3Fh] [rbp-101h] BYREF
  _BYTE *v19; // [rsp+40h] [rbp-100h] BYREF
  __int64 v20; // [rsp+48h] [rbp-F8h]
  _BYTE v21[48]; // [rsp+50h] [rbp-F0h] BYREF
  char *v22; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v23; // [rsp+88h] [rbp-B8h]
  _BYTE v24[48]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v25; // [rsp+C0h] [rbp-80h] BYREF
  char *v26; // [rsp+C8h] [rbp-78h] BYREF
  __int64 v27; // [rsp+D0h] [rbp-70h]
  _BYTE v28[104]; // [rsp+D8h] [rbp-68h] BYREF

  v2 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( !(unsigned __int8)sub_120AFE0(a1, 16, "expected ':' in memprof")
    && !(unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in memprof") )
  {
    while ( !(unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in memprof")
         && !(unsigned __int8)sub_120AFE0(a1, 292, "expected 'type' in memprof")
         && !(unsigned __int8)sub_120AFE0(a1, 16, "expected ':'")
         && !(unsigned __int8)sub_12123F0(a1, &v18)
         && !(unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in memprof")
         && !(unsigned __int8)sub_120AFE0(a1, 494, "expected 'stackIds' in memprof")
         && !(unsigned __int8)sub_120AFE0(a1, 16, "expected ':'")
         && !(unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in stackIds") )
    {
      v20 = 0xC00000000LL;
      v19 = v21;
      while ( 1 )
      {
        v5 = (__int64)&v25;
        v25 = 0;
        if ( (unsigned __int8)sub_120C050(a1, &v25) )
          goto LABEL_40;
        v6 = sub_9E27D0(*(_QWORD *)(a1 + 352), v25);
        v8 = (unsigned int)v20;
        v9 = (unsigned int)v20 + 1LL;
        if ( v9 > HIDWORD(v20) )
        {
          v17 = v6;
          sub_C8D5F0((__int64)&v19, v21, (unsigned int)v20 + 1LL, 4u, v9, v7);
          v8 = (unsigned int)v20;
          v6 = v17;
        }
        *(_DWORD *)&v19[4 * v8] = v6;
        LODWORD(v20) = v20 + 1;
        if ( *(_DWORD *)(a1 + 240) != 4 )
          break;
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
      }
      v5 = 13;
      if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' in stackIds") )
        goto LABEL_40;
      v12 = (__int64)v24;
      v23 = 0xC00000000LL;
      v22 = v24;
      if ( (_DWORD)v20 )
      {
        v16 = v18;
        sub_1205840((__int64)&v22, (__int64)&v19, v10, (__int64)v24, (__int64)&v22, v11);
        v12 = (unsigned int)v23;
        v27 = 0xC00000000LL;
        LOBYTE(v25) = v16;
        v26 = v28;
        if ( (_DWORD)v23 )
          sub_1205E10((__int64)&v26, &v22, v15, (unsigned int)v23, (__int64)&v22, v11);
      }
      else
      {
        LOBYTE(v25) = v18;
        v26 = v28;
        v27 = 0xC00000000LL;
      }
      v13 = a2[1];
      if ( v13 == a2[2] )
      {
        sub_9D3C80(a2, v13, (__int64)&v25);
      }
      else
      {
        if ( v13 )
        {
          v14 = v25;
          *(_QWORD *)(v13 + 16) = 0xC00000000LL;
          *(_BYTE *)v13 = v14;
          *(_QWORD *)(v13 + 8) = v13 + 24;
          if ( (_DWORD)v27 )
            sub_1205E10(v13 + 8, &v26, (unsigned int)v27, v12, (__int64)&v26, v11);
          v13 = a2[1];
        }
        v13 += 72;
        a2[1] = v13;
      }
      if ( v26 != v28 )
        _libc_free(v26, v13);
      if ( v22 != v24 )
        _libc_free(v22, v13);
      v5 = 13;
      if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' in memprof") )
      {
LABEL_40:
        if ( v19 != v21 )
          _libc_free(v19, v5);
        return 1;
      }
      if ( v19 != v21 )
        _libc_free(v19, 13);
      if ( *(_DWORD *)(a1 + 240) != 4 )
        return sub_120AFE0(a1, 13, "expected ')' in memprof");
      *(_DWORD *)(a1 + 240) = sub_1205200(v2);
    }
  }
  return 1;
}
