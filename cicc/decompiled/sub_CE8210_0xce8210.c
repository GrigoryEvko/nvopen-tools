// Function: sub_CE8210
// Address: 0xce8210
//
__int64 __fastcall sub_CE8210(__int64 a1)
{
  size_t v1; // rdx
  __int64 v2; // r13
  char *v3; // r12
  char *v4; // rsi
  __int64 v5; // rbx
  unsigned int v6; // r12d
  __int64 v7; // r14
  unsigned __int8 v8; // al
  char **v9; // rdi
  __int64 v10; // r13
  unsigned int v11; // r15d
  __int64 v12; // rdx
  _BYTE *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 result; // rax
  unsigned __int8 v17; // al
  unsigned __int64 v18; // rsi
  __int64 v19; // r13
  __int64 v20; // rbx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  int v26; // [rsp+20h] [rbp-D0h]
  unsigned int v27; // [rsp+24h] [rbp-CCh]
  __int64 v28; // [rsp+28h] [rbp-C8h]
  __int64 v29; // [rsp+28h] [rbp-C8h]
  _BYTE *v30; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+38h] [rbp-B8h]
  _BYTE v32[176]; // [rsp+40h] [rbp-B0h] BYREF

  v1 = 0;
  v2 = *(_QWORD *)(a1 + 40);
  v30 = v32;
  v31 = 0x1000000000LL;
  v3 = off_4C5D0E8;
  if ( off_4C5D0E8 )
    v1 = strlen(off_4C5D0E8);
  v4 = v3;
  v5 = sub_BA8DC0(v2, (__int64)v3, v1);
  if ( v5 && (v6 = 0, (v26 = sub_B91A00(v5)) != 0) )
  {
    while ( 1 )
    {
      v7 = sub_B91A10(v5, v6);
      v28 = v7 - 16;
      v8 = *(_BYTE *)(v7 - 16);
      v9 = (v8 & 2) != 0 ? *(char ***)(v7 - 32) : (char **)(v28 - 8LL * ((v8 >> 2) & 0xF));
      v4 = *v9;
      if ( *v9 )
      {
        if ( *v4 == 1 )
        {
          v4 = (char *)*((_QWORD *)v4 + 17);
          if ( (unsigned __int8)*v4 <= 3u && (char *)a1 == v4 )
          {
            if ( (v8 & 2) != 0 )
            {
              v27 = *(_DWORD *)(v7 - 24);
            }
            else
            {
              v4 = (char *)*(unsigned __int16 *)(v7 - 16);
              LOWORD(v4) = (unsigned __int16)v4 >> 6;
              v27 = (unsigned __int8)v4 & 0xF;
            }
            if ( v27 > 1 )
              break;
          }
        }
      }
LABEL_23:
      if ( v26 == ++v6 )
        goto LABEL_24;
    }
    v10 = 8;
    v11 = 1;
    if ( (v8 & 2) != 0 )
    {
LABEL_15:
      v12 = *(_QWORD *)(v7 - 32);
      goto LABEL_16;
    }
    while ( 1 )
    {
      v12 = v28 - 8LL * ((v8 >> 2) & 0xF);
LABEL_16:
      v13 = *(_BYTE **)(v12 + v10);
      if ( *v13 )
        v13 = 0;
      v14 = sub_B91420((__int64)v13);
      if ( v15 == 14
        && *(_QWORD *)v14 == 0x685F746978657461LL
        && *(_DWORD *)(v14 + 8) == 1818521185
        && *(_WORD *)(v14 + 12) == 29285 )
      {
        break;
      }
      v11 += 2;
      v10 += 16;
      if ( v27 <= v11 )
        goto LABEL_23;
      v8 = *(_BYTE *)(v7 - 16);
      if ( (v8 & 2) != 0 )
        goto LABEL_15;
    }
    v17 = *(_BYTE *)(v7 - 16);
    v18 = v11 + 1;
    if ( (v17 & 2) != 0 )
      v19 = *(_QWORD *)(v7 - 32);
    else
      v19 = v28 - 8LL * ((v17 >> 2) & 0xF);
    v20 = sub_CE7B80(v19 + 8 * v18);
    v23 = (unsigned int)v31;
    v24 = (unsigned int)v31 + 1LL;
    if ( v24 > HIDWORD(v31) )
    {
      v18 = (unsigned __int64)v32;
      sub_C8D5F0((__int64)&v30, v32, v24, 8u, v21, v22);
      v23 = (unsigned int)v31;
    }
    *(_QWORD *)&v30[8 * v23] = v20;
    LODWORD(v31) = v31 + 1;
    result = *(_QWORD *)v30;
    if ( v30 != v32 )
    {
      v29 = *(_QWORD *)v30;
      _libc_free(v30, v18);
      return v29;
    }
  }
  else
  {
LABEL_24:
    if ( v30 == v32 )
    {
      return 0;
    }
    else
    {
      _libc_free(v30, v4);
      return 0;
    }
  }
  return result;
}
