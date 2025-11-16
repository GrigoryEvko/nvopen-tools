// Function: sub_14AC610
// Address: 0x14ac610
//
unsigned __int16 *__fastcall sub_14AC610(unsigned __int16 *a1, __int64 *a2, __int64 a3)
{
  unsigned __int16 *v4; // r12
  unsigned __int16 **v6; // r8
  unsigned __int16 **v7; // rdi
  char v8; // dl
  __int64 v9; // rsi
  int v10; // eax
  int v11; // edx
  unsigned __int16 **v12; // r12
  unsigned __int16 **v13; // rsi
  unsigned __int16 **v14; // rax
  unsigned __int16 **v15; // rcx
  __int64 *v16; // r9
  __int64 v18; // rsi
  unsigned __int16 **v19; // r12
  unsigned __int16 v20; // ax
  __int64 *v21; // [rsp+0h] [rbp-100h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-F8h]
  __int64 *v23; // [rsp+10h] [rbp-F0h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-E8h]
  __int64 v25; // [rsp+20h] [rbp-E0h] BYREF
  unsigned __int16 **v26; // [rsp+28h] [rbp-D8h]
  unsigned __int16 **v27; // [rsp+30h] [rbp-D0h]
  __int64 v28; // [rsp+38h] [rbp-C8h]
  int v29; // [rsp+40h] [rbp-C0h]
  _BYTE v30[184]; // [rsp+48h] [rbp-B8h] BYREF

  v4 = a1;
  v22 = sub_15A95F0(a3, *(_QWORD *)a1);
  if ( v22 > 0x40 )
    sub_16A4EF0(&v21, 0, 0);
  else
    v21 = 0;
  v6 = (unsigned __int16 **)v30;
  v25 = 0;
  v26 = (unsigned __int16 **)v30;
  v7 = (unsigned __int16 **)v30;
  v27 = (unsigned __int16 **)v30;
  v28 = 16;
  v29 = 0;
LABEL_4:
  if ( v6 == v7 )
  {
    while ( 1 )
    {
      v13 = &v6[HIDWORD(v28)];
      if ( v13 == v6 )
      {
LABEL_55:
        if ( HIDWORD(v28) >= (unsigned int)v28 )
          goto LABEL_5;
        ++HIDWORD(v28);
        *v13 = v4;
        v6 = v26;
        ++v25;
        v7 = v27;
        goto LABEL_6;
      }
      v14 = v6;
      v15 = 0;
      do
      {
        if ( v4 == *v14 )
        {
          v16 = v21;
          if ( v22 > 0x40 )
            goto LABEL_50;
          *a2 = (__int64)((_QWORD)v21 << (64 - (unsigned __int8)v22)) >> (64 - (unsigned __int8)v22);
          return v4;
        }
        if ( *v14 == (unsigned __int16 *)-2LL )
          v15 = v14;
        ++v14;
      }
      while ( v13 != v14 );
      if ( !v15 )
        goto LABEL_55;
      *v15 = v4;
      v7 = v27;
      --v29;
      v6 = v26;
      ++v25;
      v9 = *(_QWORD *)v4;
      if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) == 16 )
        goto LABEL_44;
LABEL_7:
      v10 = *((unsigned __int8 *)v4 + 16);
      if ( (unsigned __int8)v10 > 0x17u )
        break;
      if ( (_BYTE)v10 != 5 )
      {
        if ( (_BYTE)v10 == 1 )
          __asm { jmp     rax }
        goto LABEL_44;
      }
      v20 = v4[9];
      if ( v20 == 32 )
      {
LABEL_23:
        v24 = sub_15A95F0(a3, v9);
        if ( v24 > 0x40 )
          sub_16A4EF0(&v23, 0, 0);
        else
          v23 = 0;
        if ( (unsigned __int8)sub_1634900(v4, a3, &v23) )
        {
          if ( v24 > 0x40 )
            v18 = *v23;
          else
            v18 = (__int64)((_QWORD)v23 << (64 - (unsigned __int8)v24)) >> (64 - (unsigned __int8)v24);
          sub_16A7490(&v21, v18);
          if ( (*((_BYTE *)v4 + 23) & 0x40) != 0 )
            v19 = (unsigned __int16 **)*((_QWORD *)v4 - 1);
          else
            v19 = (unsigned __int16 **)&v4[-12 * (*((_DWORD *)v4 + 5) & 0xFFFFFFF)];
          v4 = *v19;
          if ( v24 > 0x40 && v23 )
            j_j___libc_free_0_0(v23);
          v7 = v27;
          v6 = v26;
          goto LABEL_4;
        }
        if ( v24 > 0x40 && v23 )
          j_j___libc_free_0_0(v23);
        v7 = v27;
        v6 = v26;
        goto LABEL_44;
      }
      v11 = v20;
      if ( v20 != 47 )
        goto LABEL_10;
      if ( (*((_BYTE *)v4 + 23) & 0x40) == 0 )
      {
LABEL_38:
        v12 = (unsigned __int16 **)&v4[-12 * (*((_DWORD *)v4 + 5) & 0xFFFFFFF)];
        goto LABEL_13;
      }
LABEL_12:
      v12 = (unsigned __int16 **)*((_QWORD *)v4 - 1);
LABEL_13:
      v4 = *v12;
      if ( v6 != v7 )
        goto LABEL_5;
    }
    if ( (_BYTE)v10 == 56 )
      goto LABEL_23;
    v11 = v10 - 24;
    if ( v10 != 71 )
    {
LABEL_10:
      if ( v11 != 48 )
        goto LABEL_44;
    }
    if ( (*((_BYTE *)v4 + 23) & 0x40) == 0 )
      goto LABEL_38;
    goto LABEL_12;
  }
LABEL_5:
  sub_16CCBA0(&v25, v4);
  v7 = v27;
  v6 = v26;
  if ( v8 )
  {
LABEL_6:
    v9 = *(_QWORD *)v4;
    if ( *(_BYTE *)(*(_QWORD *)v4 + 8LL) != 16 )
      goto LABEL_7;
  }
LABEL_44:
  v16 = v21;
  if ( v22 <= 0x40 )
  {
    *a2 = (__int64)((_QWORD)v21 << (64 - (unsigned __int8)v22)) >> (64 - (unsigned __int8)v22);
    if ( v6 == v7 )
      return v4;
    _libc_free((unsigned __int64)v7);
    if ( v22 <= 0x40 )
      return v4;
    goto LABEL_47;
  }
LABEL_50:
  *a2 = *v16;
  if ( v6 != v7 )
  {
    _libc_free((unsigned __int64)v7);
    if ( v22 <= 0x40 )
      return v4;
LABEL_47:
    v16 = v21;
  }
  if ( v16 )
    j_j___libc_free_0_0(v16);
  return v4;
}
