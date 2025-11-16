// Function: sub_AC23B0
// Address: 0xac23b0
//
__int64 __fastcall sub_AC23B0(__int64 a1, __int64 (__fastcall *a2)(__int64))
{
  __int64 result; // rax
  _QWORD **v4; // rsi
  _QWORD *v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // r15
  _QWORD *v9; // r14
  _BYTE *v10; // rbx
  __int64 *v11; // rax
  __int64 *v12; // rdx
  _QWORD *v13; // rdi
  char v14; // dl
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int8 v17; // [rsp+8h] [rbp-E8h]
  unsigned __int8 v18; // [rsp+8h] [rbp-E8h]
  _QWORD *v19; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v20; // [rsp+18h] [rbp-D8h]
  _QWORD v21[8]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v22; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v23; // [rsp+68h] [rbp-88h]
  __int64 v24; // [rsp+70h] [rbp-80h]
  int v25; // [rsp+78h] [rbp-78h]
  char v26; // [rsp+7Ch] [rbp-74h]
  __int64 v27; // [rsp+80h] [rbp-70h] BYREF

  LODWORD(result) = 1;
  v20 = 0x800000001LL;
  v24 = 0x100000008LL;
  v4 = &v19;
  v19 = v21;
  v21[0] = a1;
  v25 = 0;
  v26 = 1;
  v27 = a1;
  v22 = 1;
  v23 = &v27;
  v5 = v21;
  while ( 1 )
  {
    v6 = v5[(unsigned int)result - 1];
    LODWORD(v20) = result - 1;
    if ( *(_BYTE *)v6 <= 3u )
    {
      result = a2(v6);
      if ( (_BYTE)result )
        break;
    }
    v7 = 4LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
    {
      v8 = *(_QWORD **)(v6 - 8);
      v9 = &v8[v7];
    }
    else
    {
      v9 = (_QWORD *)v6;
      v8 = (_QWORD *)(v6 - v7 * 8);
    }
    if ( v8 != v9 )
    {
      while ( 1 )
      {
        v10 = (_BYTE *)*v8;
        if ( *(_BYTE *)*v8 > 0x15u )
          goto LABEL_13;
        if ( v26 )
        {
          v11 = v23;
          v12 = &v23[HIDWORD(v24)];
          if ( v23 != v12 )
          {
            while ( v10 != (_BYTE *)*v11 )
            {
              if ( v12 == ++v11 )
                goto LABEL_24;
            }
            goto LABEL_13;
          }
LABEL_24:
          if ( HIDWORD(v24) < (unsigned int)v24 )
          {
            ++HIDWORD(v24);
            *v12 = (__int64)v10;
            ++v22;
            goto LABEL_20;
          }
        }
        v4 = (_QWORD **)*v8;
        sub_C8CC70(&v22, *v8);
        if ( v14 )
        {
LABEL_20:
          v15 = (unsigned int)v20;
          v16 = (unsigned int)v20 + 1LL;
          if ( v16 > HIDWORD(v20) )
          {
            v4 = (_QWORD **)v21;
            sub_C8D5F0(&v19, v21, v16, 8);
            v15 = (unsigned int)v20;
          }
          v8 += 4;
          v19[v15] = v10;
          LODWORD(v20) = v20 + 1;
          if ( v9 == v8 )
            break;
        }
        else
        {
LABEL_13:
          v8 += 4;
          if ( v9 == v8 )
            break;
        }
      }
    }
    v13 = v19;
    result = (unsigned int)v20;
    v5 = v19;
    if ( !(_DWORD)v20 )
      goto LABEL_15;
  }
  v13 = v19;
LABEL_15:
  if ( v13 != v21 )
  {
    v17 = result;
    _libc_free(v13, v4);
    result = v17;
  }
  if ( !v26 )
  {
    v18 = result;
    _libc_free(v23, v4);
    return v18;
  }
  return result;
}
