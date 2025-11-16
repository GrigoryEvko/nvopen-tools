// Function: sub_AD8AC0
// Address: 0xad8ac0
//
unsigned __int8 *__fastcall sub_AD8AC0(__int64 a1)
{
  _BYTE *v1; // r14
  __int64 v3; // r12
  unsigned int v4; // ebx
  int v5; // eax
  unsigned __int8 *result; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // r14d
  unsigned int v10; // ebx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // r8
  __int64 v15; // rsi
  int v16; // ecx
  unsigned int v17; // r15d
  unsigned __int64 *v18; // rdi
  unsigned __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rdx
  _BYTE *v22; // rax
  __int64 *v23; // rdi
  __int64 v24; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v25; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v26; // [rsp+8h] [rbp-68h]
  __int64 *v27; // [rsp+10h] [rbp-60h] BYREF
  __int64 v28; // [rsp+18h] [rbp-58h]
  _BYTE v29[80]; // [rsp+20h] [rbp-50h] BYREF

  v1 = (_BYTE *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)a1 == 17 )
    goto LABEL_2;
  v8 = *(unsigned __int8 *)(v3 + 8);
  if ( (unsigned int)(v8 - 17) <= 1 )
  {
    v22 = sub_AD7630(a1, 0, v8);
    if ( !v22 || (v1 = v22 + 24, *v22 != 17) )
    {
LABEL_7:
      LOBYTE(v8) = *(_BYTE *)(v3 + 8);
      goto LABEL_8;
    }
LABEL_2:
    v4 = *((_DWORD *)v1 + 2);
    if ( v4 <= 0x40 )
    {
      v7 = *(_QWORD *)v1;
      if ( *(_QWORD *)v1 && (v7 & (v7 - 1)) == 0 )
      {
        _BitScanReverse64(&v7, v7);
        v5 = v4 + (v7 ^ 0x3F) - 64;
        return (unsigned __int8 *)sub_AD64C0(v3, v4 - 1 - v5, 0);
      }
    }
    else if ( (unsigned int)sub_C44630(v1) == 1 )
    {
      v5 = sub_C444A0(v1);
      return (unsigned __int8 *)sub_AD64C0(v3, v4 - 1 - v5, 0);
    }
    goto LABEL_7;
  }
LABEL_8:
  result = 0;
  if ( (_BYTE)v8 == 17 )
  {
    v9 = *(_DWORD *)(v3 + 32);
    v27 = (__int64 *)v29;
    v28 = 0x400000000LL;
    if ( v9 )
    {
      v10 = 0;
      while ( 1 )
      {
        v15 = v10;
        result = (unsigned __int8 *)sub_AD69F0((unsigned __int8 *)a1, v10);
        if ( !result )
          goto LABEL_30;
        v16 = *result;
        if ( (unsigned int)(v16 - 12) <= 1 )
        {
          v11 = v3;
          if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
            v11 = **(_QWORD **)(v3 + 16);
          v12 = sub_AD6530(v11, v10);
          v13 = (unsigned int)v28;
          v14 = (unsigned int)v28 + 1LL;
          if ( v14 <= HIDWORD(v28) )
            goto LABEL_14;
        }
        else
        {
          if ( (_BYTE)v16 != 17 )
          {
            v21 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)result + 1) + 8LL) - 17;
            if ( (unsigned int)v21 > 1
              || (v15 = 0, (result = sub_AD7630((__int64)result, 0, v21)) == 0)
              || *result != 17 )
            {
LABEL_29:
              result = 0;
              goto LABEL_30;
            }
          }
          v17 = *((_DWORD *)result + 8);
          v18 = (unsigned __int64 *)(result + 24);
          if ( v17 > 0x40 )
          {
            v26 = result + 24;
            if ( (unsigned int)sub_C44630(v18) != 1 )
              goto LABEL_29;
            LODWORD(v19) = sub_C444A0(v26);
          }
          else
          {
            v19 = *v18;
            if ( !*v18 || (v19 & (v19 - 1)) != 0 )
              goto LABEL_29;
            _BitScanReverse64(&v19, v19);
            LODWORD(v19) = v17 + (v19 ^ 0x3F) - 64;
          }
          v20 = v3;
          if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
            v20 = **(_QWORD **)(v3 + 16);
          v12 = sub_AD64C0(v20, v17 - 1 - (unsigned int)v19, 0);
          v13 = (unsigned int)v28;
          v14 = (unsigned int)v28 + 1LL;
          if ( v14 <= HIDWORD(v28) )
            goto LABEL_14;
        }
        v24 = v12;
        sub_C8D5F0(&v27, v29, v14, 8);
        v13 = (unsigned int)v28;
        v12 = v24;
LABEL_14:
        ++v10;
        v27[v13] = v12;
        LODWORD(v28) = v28 + 1;
        if ( v9 == v10 )
        {
          v23 = v27;
          v15 = (unsigned int)v28;
          goto LABEL_42;
        }
      }
    }
    v23 = (__int64 *)v29;
    v15 = 0;
LABEL_42:
    result = (unsigned __int8 *)sub_AD3730(v23, v15);
LABEL_30:
    if ( v27 != (__int64 *)v29 )
    {
      v25 = result;
      _libc_free(v27, v15);
      return v25;
    }
  }
  return result;
}
