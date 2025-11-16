// Function: sub_D9B130
// Address: 0xd9b130
//
__int64 __fastcall sub_D9B130(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  __int64 v4; // rsi
  unsigned int v5; // r12d
  char v6; // cl
  __int64 *v8; // rax
  __int64 *v9; // r13
  __int64 *v10; // r12
  char *v11; // rax
  char *v12; // rdx
  __int64 *v13; // rax
  __int64 *v14; // rax
  __int64 v15; // [rsp+0h] [rbp-80h] BYREF
  __int64 *v16; // [rsp+8h] [rbp-78h]
  __int64 v17; // [rsp+10h] [rbp-70h]
  unsigned int v18; // [rsp+18h] [rbp-68h]
  char v19; // [rsp+1Ch] [rbp-64h]
  char v20; // [rsp+20h] [rbp-60h] BYREF
  __int64 v21; // [rsp+30h] [rbp-50h] BYREF
  char *v22; // [rsp+38h] [rbp-48h]
  __int64 v23; // [rsp+40h] [rbp-40h]
  int v24; // [rsp+48h] [rbp-38h]
  char v25; // [rsp+4Ch] [rbp-34h]
  char v26; // [rsp+50h] [rbp-30h] BYREF

  v16 = (__int64 *)&v20;
  v22 = &v26;
  v15 = 0;
  v17 = 2;
  v18 = 0;
  v19 = 1;
  v21 = 0;
  v23 = 2;
  v24 = 0;
  v25 = 1;
  sub_D9A5D0(a1, a2, (__int64)&v15);
  sub_D9A5D0(a1, a3, (__int64)&v21);
  v4 = v18;
  if ( v18 == HIDWORD(v17) )
  {
    v6 = v25;
    v5 = 1;
    goto LABEL_4;
  }
  v5 = 1;
  v6 = v25;
  if ( HIDWORD(v23) == v24 || (v5 = 0, HIDWORD(v23) - v24 != HIDWORD(v17) - v18) )
  {
LABEL_4:
    if ( v6 )
      goto LABEL_5;
LABEL_13:
    _libc_free(v22, v4);
    if ( v19 )
      return v5;
    goto LABEL_14;
  }
  v8 = v16;
  if ( v19 )
  {
    v9 = &v16[HIDWORD(v17)];
    if ( v16 != v9 )
      goto LABEL_9;
  }
  else
  {
    v9 = &v16[(unsigned int)v17];
    if ( v16 != v9 )
    {
LABEL_9:
      while ( 1 )
      {
        v4 = *v8;
        v10 = v8;
        if ( (unsigned __int64)*v8 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v9 == ++v8 )
          goto LABEL_11;
      }
      while ( v9 != v10 )
      {
        if ( v6 )
        {
          v11 = v22;
          v12 = &v22[8 * HIDWORD(v23)];
          if ( v22 == v12 )
            goto LABEL_26;
          while ( v4 != *(_QWORD *)v11 )
          {
            v11 += 8;
            if ( v12 == v11 )
              goto LABEL_26;
          }
        }
        else
        {
          v14 = sub_C8CA60((__int64)&v21, v4);
          v6 = v25;
          if ( !v14 )
          {
LABEL_26:
            v5 = 0;
            goto LABEL_4;
          }
        }
        v13 = v10 + 1;
        if ( v10 + 1 == v9 )
          break;
        while ( 1 )
        {
          v4 = *v13;
          v10 = v13;
          if ( (unsigned __int64)*v13 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v9 == ++v13 )
          {
            v5 = 1;
            goto LABEL_12;
          }
        }
      }
    }
  }
LABEL_11:
  v5 = 1;
LABEL_12:
  if ( !v6 )
    goto LABEL_13;
LABEL_5:
  if ( v19 )
    return v5;
LABEL_14:
  _libc_free(v16, v4);
  return v5;
}
