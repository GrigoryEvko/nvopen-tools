// Function: sub_15CD9D0
// Address: 0x15cd9d0
//
void __fastcall sub_15CD9D0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 *v4; // r13
  __int64 *v6; // r15
  unsigned int v7; // ecx
  char *v8; // rdi
  __int64 v9; // r14
  __int64 v10; // r9
  __int64 v11; // rax
  char *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rax
  char *v15; // rdx
  char *v16; // rax
  __int64 v17; // [rsp+0h] [rbp-D0h]
  char *v18; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v19; // [rsp+18h] [rbp-B8h]
  _BYTE v20[176]; // [rsp+20h] [rbp-B0h] BYREF

  v3 = 2 * a3;
  v4 = &a2[v3];
  v18 = v20;
  v19 = 0x800000000LL;
  if ( &a2[v3] != a2 )
  {
    v6 = a2;
    v7 = 0;
    v8 = v20;
    while ( 1 )
    {
      v9 = *v6;
      v10 = v6[1];
      v11 = 16LL * v7;
      v12 = &v8[v11];
      v13 = v11 >> 4;
      v14 = v11 >> 6;
      if ( v14 )
      {
        v15 = &v8[64 * v14];
        v16 = v8;
        while ( *((_QWORD *)v16 + 1) != v10 || *(_QWORD *)v16 != v9 )
        {
          if ( *((_QWORD *)v16 + 2) == v9 && *((_QWORD *)v16 + 3) == v10 )
          {
            v16 += 16;
            goto LABEL_24;
          }
          if ( *((_QWORD *)v16 + 4) == v9 && *((_QWORD *)v16 + 5) == v10 )
          {
            v16 += 32;
            goto LABEL_24;
          }
          if ( *((_QWORD *)v16 + 6) == v9 && *((_QWORD *)v16 + 7) == v10 )
          {
            v16 += 48;
            goto LABEL_24;
          }
          v16 += 64;
          if ( v15 == v16 )
          {
            v13 = (v12 - v16) >> 4;
            goto LABEL_15;
          }
        }
        goto LABEL_24;
      }
      v16 = v8;
LABEL_15:
      if ( v13 == 2 )
        goto LABEL_32;
      if ( v13 == 3 )
        break;
      if ( v13 != 1 )
        goto LABEL_18;
LABEL_35:
      if ( *((_QWORD *)v16 + 1) != v10 )
      {
LABEL_18:
        if ( HIDWORD(v19) <= v7 )
          goto LABEL_38;
        goto LABEL_19;
      }
      if ( *(_QWORD *)v16 != v9 )
      {
        if ( HIDWORD(v19) <= v7 )
        {
LABEL_38:
          v17 = v6[1];
          sub_16CD150(&v18, v20, 0, 16);
          v10 = v17;
          v12 = &v18[16 * (unsigned int)v19];
        }
LABEL_19:
        *(_QWORD *)v12 = v9;
        *((_QWORD *)v12 + 1) = v10;
        v6 += 2;
        LODWORD(v19) = v19 + 1;
        sub_15CD750(a1, (v10 >> 2) & 1, v9, v10 & 0xFFFFFFFFFFFFFFF8LL);
        v8 = v18;
        if ( v4 == v6 )
          goto LABEL_20;
        goto LABEL_26;
      }
LABEL_24:
      if ( v12 == v16 )
        goto LABEL_18;
      v6 += 2;
      if ( v4 == v6 )
      {
LABEL_20:
        if ( v8 != v20 )
          _libc_free((unsigned __int64)v8);
        return;
      }
LABEL_26:
      v7 = v19;
    }
    if ( *(_QWORD *)v16 == v9 && *((_QWORD *)v16 + 1) == v10 )
      goto LABEL_24;
    v16 += 16;
LABEL_32:
    if ( *((_QWORD *)v16 + 1) == v10 && *(_QWORD *)v16 == v9 )
      goto LABEL_24;
    v16 += 16;
    goto LABEL_35;
  }
}
