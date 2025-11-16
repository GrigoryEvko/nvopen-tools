// Function: sub_AA9140
// Address: 0xaa9140
//
__int64 __fastcall sub_AA9140(__int64 a1)
{
  __int64 v2; // rsi
  char v3; // r13
  _BYTE *v4; // rdi
  unsigned int v5; // r12d
  unsigned int v7; // r13d
  bool v8; // al
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned int v11; // ebx
  int v12; // r13d
  char v13; // r14
  unsigned int v14; // r15d
  __int64 v15; // rax
  unsigned int v16; // r14d
  _QWORD v17[2]; // [rsp+0h] [rbp-F0h] BYREF
  _BYTE *v18; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v19; // [rsp+18h] [rbp-D8h]
  _BYTE v20[64]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v21; // [rsp+60h] [rbp-90h] BYREF
  char *v22; // [rsp+68h] [rbp-88h]
  __int64 v23; // [rsp+70h] [rbp-80h]
  int v24; // [rsp+78h] [rbp-78h]
  char v25; // [rsp+7Ch] [rbp-74h]
  char v26; // [rsp+80h] [rbp-70h] BYREF

  if ( (unsigned __int8)(*(_BYTE *)a1 - 12) <= 1u )
    return 1;
  if ( (unsigned __int8)(*(_BYTE *)a1 - 9) <= 2u )
  {
    v2 = a1;
    v22 = &v26;
    v23 = 8;
    v21 = 0;
    v24 = 0;
    v25 = 1;
    v18 = v20;
    v19 = 0x800000000LL;
    v17[0] = &v21;
    v17[1] = &v18;
    v3 = sub_AA8FD0(v17, a1);
    if ( v3 )
    {
      while ( 1 )
      {
        v4 = v18;
        if ( !(_DWORD)v19 )
          break;
        v2 = *(_QWORD *)&v18[8 * (unsigned int)v19 - 8];
        LODWORD(v19) = v19 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(v17, v2) )
          goto LABEL_20;
      }
    }
    else
    {
LABEL_20:
      v4 = v18;
      v3 = 0;
    }
    if ( v4 != v20 )
      _libc_free(v4, v2);
    if ( !v25 )
      _libc_free(v22, v2);
    if ( v3 )
      return 1;
  }
  v5 = sub_AC30F0(a1);
  if ( (_BYTE)v5 )
    return 1;
  if ( *(_BYTE *)a1 == 17 )
  {
    v7 = *(_DWORD *)(a1 + 32);
    if ( v7 <= 0x40 )
      v8 = *(_QWORD *)(a1 + 24) == 0;
    else
      v8 = v7 == (unsigned int)sub_C444A0(a1 + 24);
    goto LABEL_18;
  }
  v9 = *(_QWORD *)(a1 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 > 1 )
    return v5;
  v10 = sub_AD7630(a1, 0);
  if ( v10 && *(_BYTE *)v10 == 17 )
  {
    v11 = *(_DWORD *)(v10 + 32);
    if ( v11 <= 0x40 )
    {
      if ( *(_QWORD *)(v10 + 24) )
        return v5;
      return 1;
    }
    v8 = v11 == (unsigned int)sub_C444A0(v10 + 24);
LABEL_18:
    if ( !v8 )
      return v5;
    return 1;
  }
  if ( *(_BYTE *)(v9 + 8) == 17 )
  {
    v12 = *(_DWORD *)(v9 + 32);
    if ( v12 )
    {
      v13 = 0;
      v14 = 0;
      while ( 1 )
      {
        v15 = sub_AD69F0(a1, v14);
        if ( !v15 )
          break;
        if ( *(_BYTE *)v15 != 13 )
        {
          if ( *(_BYTE *)v15 != 17 )
            return v5;
          v16 = *(_DWORD *)(v15 + 32);
          if ( v16 <= 0x40 )
          {
            if ( *(_QWORD *)(v15 + 24) )
              return v5;
          }
          else if ( v16 != (unsigned int)sub_C444A0(v15 + 24) )
          {
            return v5;
          }
          v13 = 1;
        }
        if ( v12 == ++v14 )
        {
          if ( v13 )
            return 1;
          return v5;
        }
      }
    }
  }
  return v5;
}
