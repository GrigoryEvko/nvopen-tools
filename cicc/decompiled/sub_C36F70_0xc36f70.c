// Function: sub_C36F70
// Address: 0xc36f70
//
__int64 __fastcall sub_C36F70(_DWORD **a1, _BYTE *a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // rax
  _BYTE *v5; // rcx
  char v6; // r13
  char v7; // bl
  _BYTE *v8; // r14
  unsigned int v9; // r8d
  unsigned __int64 v11; // rdx
  _BYTE *v12; // rsi
  char v13; // di
  __int64 v14; // rsi
  _BYTE *v15; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v16; // [rsp+18h] [rbp-48h]
  unsigned __int64 v17; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-38h]

  v15 = a2;
  v16 = a3;
  if ( a3 <= 2 )
    return 0;
  v4 = a3;
  v5 = a2;
  switch ( a3 )
  {
    case 3uLL:
      if ( *(_WORD *)a2 == 28265 && a2[2] == 102 )
        goto LABEL_34;
      break;
    case 8uLL:
      if ( *(_QWORD *)a2 != 0x5954494E49464E49LL )
      {
        if ( *a2 != 45 )
        {
          if ( (*a2 & 0xDF) != 0x53 )
          {
LABEL_7:
            v6 = 0;
            v7 = 0;
            goto LABEL_8;
          }
          v5 = a2 + 1;
          v16 = 7;
          v6 = 0;
          v4 = 7;
          v15 = a2 + 1;
          goto LABEL_32;
        }
        v16 = 7;
        v15 = a2 + 1;
        goto LABEL_24;
      }
LABEL_34:
      sub_C36EF0(a1, 0);
      return 1;
    case 4uLL:
      if ( *(_DWORD *)a2 == 1718503723 )
        goto LABEL_34;
      if ( *a2 != 45 )
      {
        if ( (*a2 & 0xDF) != 0x53 )
          goto LABEL_7;
        v5 = a2 + 1;
        v16 = 3;
        v6 = 0;
        v4 = 3;
        v15 = a2 + 1;
        goto LABEL_32;
      }
      v12 = a2 + 1;
      v16 = 3;
      v15 = v12;
      goto LABEL_36;
  }
  if ( *a2 == 45 )
  {
    v11 = a3 - 1;
    v12 = a2 + 1;
    v15 = v12;
    v16 = v11;
    if ( v11 == 2 )
      return 0;
    if ( v11 != 3 )
    {
      if ( v11 == 8 && *(_QWORD *)v12 == 0x5954494E49464E49LL )
      {
LABEL_39:
        sub_C36EF0(a1, 1);
        return 1;
      }
LABEL_24:
      v5 = v15;
      v4 = v16;
      if ( (*v15 & 0xDF) != 0x53 )
      {
        if ( v16 > 2 )
        {
          v6 = 1;
          v7 = 0;
          goto LABEL_8;
        }
        return 0;
      }
      v6 = 1;
      goto LABEL_15;
    }
LABEL_36:
    if ( *(_WORD *)v12 == 28265 && v12[2] == 102 || *(_WORD *)v12 == 28233 && v12[2] == 102 )
      goto LABEL_39;
    goto LABEL_24;
  }
  if ( (*a2 & 0xDF) != 0x53 )
    goto LABEL_7;
  v6 = 0;
LABEL_15:
  --v4;
  v15 = ++v5;
  v16 = v4;
  if ( v4 <= 2 )
    return 0;
LABEL_32:
  v7 = 1;
LABEL_8:
  if ( (*(_WORD *)v5 != 24942 || v5[2] != 110) && (*(_WORD *)v5 != 24910 || v5[2] != 78) )
    return 0;
  v8 = v5 + 3;
  v15 = v5 + 3;
  v16 = v4 - 3;
  if ( v4 == 3 )
  {
    sub_C36070((__int64)a1, v7, v6, 0);
    return 1;
  }
  v13 = v5[3];
  if ( v13 == 40 )
  {
    v9 = 0;
    if ( v4 - 3 <= 2 || v8[v4 - 4] != 41 )
      return v9;
    v8 = v5 + 4;
    v15 = v5 + 4;
    v16 = v4 - 5;
    v13 = v5[4];
  }
  v14 = 10;
  if ( v13 == 48 )
  {
    v14 = 8;
    if ( v16 > 1 )
    {
      v14 = 8;
      if ( tolower((char)v8[1]) == 120 )
      {
        v14 = 16;
        v15 = v8 + 2;
        v16 -= 2LL;
      }
    }
  }
  v18 = 1;
  v17 = 0;
  if ( (unsigned __int8)sub_C94210(&v15, v14, &v17) )
  {
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    return 0;
  }
  sub_C36070((__int64)a1, v7, v6, &v17);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  return 1;
}
