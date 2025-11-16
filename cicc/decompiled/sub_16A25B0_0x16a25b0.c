// Function: sub_16A25B0
// Address: 0x16a25b0
//
__int64 __fastcall sub_16A25B0(__int64 a1, __int64 a2)
{
  __int16 *v2; // rax
  __int16 *v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // r9
  _QWORD *v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // [rsp+8h] [rbp-B8h]
  _QWORD *v11; // [rsp+10h] [rbp-B0h]
  _QWORD *v12; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v13; // [rsp+1Fh] [rbp-A1h]
  __int64 v14; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-98h]
  _QWORD v16[3]; // [rsp+38h] [rbp-88h] BYREF
  _BYTE v17[8]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v18[3]; // [rsp+58h] [rbp-68h] BYREF
  __int64 v19; // [rsp+70h] [rbp-50h] BYREF
  _QWORD v20[9]; // [rsp+78h] [rbp-48h] BYREF

  sub_169D930((__int64)&v19, a1);
  v2 = (__int16 *)sub_16982C0();
  v3 = v2;
  if ( v2 == word_42AE980 )
    sub_169D060(v16, (__int64)v2, &v19);
  else
    sub_169D050((__int64)v16, word_42AE980, &v19);
  if ( LODWORD(v20[0]) > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( !a2 )
  {
    if ( v3 == (__int16 *)v16[0] )
      v13 = sub_16A25B0(v16, 0);
    else
      v13 = sub_16A0030((__int64)v16, 0);
    goto LABEL_21;
  }
  if ( v3 == word_42AE980 )
  {
    sub_169C4E0(v18, (__int64)v3);
    if ( v3 != (__int16 *)v16[0] )
      goto LABEL_9;
  }
  else
  {
    sub_1698360((__int64)v18, (__int64)word_42AE980);
    if ( v3 != (__int16 *)v16[0] )
    {
LABEL_9:
      v13 = sub_16A0030((__int64)v16, (__int64)v17);
      goto LABEL_10;
    }
  }
  v13 = sub_16A25B0(v16, v17);
LABEL_10:
  if ( v3 == (__int16 *)v18[0] )
    sub_169D930((__int64)&v14, (__int64)v18);
  else
    sub_169D7E0((__int64)&v14, v18);
  if ( v3 == (__int16 *)&unk_42AE990 )
    sub_169D060(v20, (__int64)v3, &v14);
  else
    sub_169D050((__int64)v20, &unk_42AE990, &v14);
  v4 = v20[0];
  v5 = (_QWORD *)(a2 + 8);
  if ( v3 == *(__int16 **)(a2 + 8) )
  {
    if ( v3 == (__int16 *)v20[0] )
    {
      v8 = *(_QWORD *)(a2 + 16);
      if ( v8 )
      {
        v9 = v8 + 32LL * *(_QWORD *)(v8 - 8);
        while ( v8 != v9 )
        {
          v9 -= 32;
          v10 = v4;
          v11 = v5;
          if ( *(_QWORD *)(v9 + 8) == v4 )
          {
            sub_169DEB0((__int64 *)(v9 + 16));
            v4 = v10;
            v5 = v11;
          }
          else
          {
            sub_1698460(v9 + 8);
            v5 = v11;
            v4 = v10;
          }
        }
        v12 = v5;
        j_j_j___libc_free_0_0(v8 - 8);
        v5 = v12;
      }
      sub_169C7E0(v5, v20);
      goto LABEL_17;
    }
    goto LABEL_25;
  }
  if ( v3 == (__int16 *)v20[0] )
  {
LABEL_25:
    sub_127D120((_QWORD *)(a2 + 8));
    v7 = (_QWORD *)(a2 + 8);
    if ( v3 == (__int16 *)v20[0] )
      sub_169C7E0(v7, v20);
    else
      sub_1698450((__int64)v7, (__int64)v20);
    goto LABEL_17;
  }
  sub_16983E0(a2 + 8, (__int64)v20);
LABEL_17:
  sub_127D120(v20);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  sub_127D120(v18);
LABEL_21:
  sub_127D120(v16);
  return v13;
}
