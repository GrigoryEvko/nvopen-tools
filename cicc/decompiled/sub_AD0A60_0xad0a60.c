// Function: sub_AD0A60
// Address: 0xad0a60
//
__int64 __fastcall sub_AD0A60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r10
  __int64 *v6; // r11
  __int64 *v7; // rbx
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // r12
  unsigned __int64 v12; // rcx
  int v13; // r8d
  __int64 v14; // rsi
  __int64 v15; // rsi
  _QWORD *v16; // rax
  __int64 *v17; // rsi
  __int64 v18; // r12
  __int64 v20; // [rsp+8h] [rbp-88h]
  __int64 *v21; // [rsp+10h] [rbp-80h]
  __int64 *v22; // [rsp+18h] [rbp-78h]
  __int64 *v23; // [rsp+18h] [rbp-78h]
  int v24; // [rsp+20h] [rbp-70h]
  __int64 v25; // [rsp+20h] [rbp-70h]
  __int64 *v26; // [rsp+28h] [rbp-68h]
  int v27; // [rsp+28h] [rbp-68h]
  __int64 *v28; // [rsp+30h] [rbp-60h] BYREF
  __int64 v29; // [rsp+38h] [rbp-58h]
  _BYTE v30[80]; // [rsp+40h] [rbp-50h] BYREF

  v4 = (__int64 *)v30;
  v28 = (__int64 *)v30;
  v29 = 0x400000000LL;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v6 = *(__int64 **)(a1 - 8);
  else
    v6 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v7 = v6;
  v8 = v6 + 16;
  v9 = 0;
  LODWORD(v10) = 0;
  v11 = *v6;
  v12 = 4;
  v13 = 0;
  if ( *v6 == a2 )
    goto LABEL_7;
LABEL_4:
  v14 = v9 + 1;
  if ( v9 + 1 > v12 )
    goto LABEL_8;
  while ( 1 )
  {
    v7 += 4;
    v28[v9] = v11;
    v15 = (unsigned int)v29;
    v9 = (unsigned int)(v29 + 1);
    LODWORD(v29) = v29 + 1;
    if ( v8 == v7 )
      break;
    v11 = *v7;
    v12 = HIDWORD(v29);
    if ( *v7 != a2 )
      goto LABEL_4;
LABEL_7:
    v14 = v9 + 1;
    ++v13;
    v11 = a3;
    v10 = ((char *)v7 - (char *)v6) >> 5;
    if ( v9 + 1 > v12 )
    {
LABEL_8:
      v20 = a3;
      v21 = v6;
      v22 = v8;
      v24 = v13;
      v26 = v4;
      sub_C8D5F0(&v28, v4, v14, 8);
      v9 = (unsigned int)v29;
      a3 = v20;
      v6 = v21;
      v8 = v22;
      v13 = v24;
      v4 = v26;
    }
  }
  v23 = v4;
  v25 = a3;
  v27 = v13;
  v16 = (_QWORD *)sub_BD5C60(a1, v15, v9);
  v17 = v28;
  v18 = sub_AD0760(*v16 + 2088LL, v28, (unsigned int)v29, (__int64 *)a1, a2, v25, v27, v10);
  if ( v28 != v23 )
    _libc_free(v28, v17);
  return v18;
}
