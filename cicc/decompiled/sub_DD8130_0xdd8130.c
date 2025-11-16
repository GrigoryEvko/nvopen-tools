// Function: sub_DD8130
// Address: 0xdd8130
//
__int64 __fastcall sub_DD8130(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // rdx
  unsigned int v4; // eax
  __int64 v5; // r13
  unsigned __int64 v6; // r15
  unsigned __int64 v7; // rsi
  __int64 *v8; // rdx
  __int64 v9; // r9
  _BYTE *v10; // rdi
  __int64 v11; // r12
  __int64 v13; // rdx
  unsigned int v14; // eax
  unsigned __int64 *v15; // rdx
  __int64 v16; // r8
  _QWORD *v17; // rax
  __int64 v18; // rcx
  unsigned int v19; // edx
  unsigned __int64 v20; // r13
  __int64 v21; // r15
  _QWORD *v22; // [rsp+0h] [rbp-E0h]
  __int64 v23; // [rsp+8h] [rbp-D8h]
  _QWORD *v25; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-A8h]
  unsigned int v27; // [rsp+3Ch] [rbp-A4h]
  _QWORD v28[6]; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE *v29; // [rsp+70h] [rbp-70h] BYREF
  __int64 v30; // [rsp+78h] [rbp-68h]
  _BYTE v31[96]; // [rsp+80h] [rbp-60h] BYREF

  v3 = v28;
  v25 = v28;
  v28[1] = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v4 = 2;
  v27 = 6;
  v28[0] = a2 | 4;
  v26 = 2;
  while ( 1 )
  {
    v5 = v3[v4 - 1];
    v26 = v4 - 1;
    v6 = v5 & 0xFFFFFFFFFFFFFFF8LL;
    if ( sub_D98300((__int64)a1, v5 & 0xFFFFFFFFFFFFFFF8LL) )
      goto LABEL_2;
    v30 = 0x600000000LL;
    v29 = v31;
    if ( (v5 & 4) != 0 )
    {
      v7 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      v8 = sub_DD80F0(a1, v6);
      if ( v8 )
        goto LABEL_7;
    }
    else
    {
      v7 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      v8 = sub_DA3A30((__int64)a1, (unsigned __int8 *)v6, (__int64)&v29);
      if ( v8 )
      {
LABEL_7:
        v7 = v5 & 0xFFFFFFFFFFFFFFF8LL;
        sub_DB77A0((__int64)a1, v6, (__int64)v8);
        v10 = v29;
        goto LABEL_8;
      }
    }
    v13 = v26;
    v14 = v26;
    if ( v26 >= (unsigned __int64)v27 )
    {
      v21 = v6 | 4;
      if ( v27 < (unsigned __int64)v26 + 1 )
      {
        v7 = (unsigned __int64)v28;
        sub_C8D5F0((__int64)&v25, v28, v26 + 1LL, 8u, v26 + 1LL, v9);
        v13 = v26;
      }
      v25[v13] = v21;
      ++v26;
    }
    else
    {
      v15 = &v25[v26];
      if ( v15 )
      {
        v7 = v6 | 4;
        *v15 = v6 | 4;
        v14 = v26;
      }
      v26 = v14 + 1;
    }
    v10 = v29;
    v16 = (__int64)&v29[8 * (unsigned int)v30];
    if ( (_BYTE *)v16 != v29 )
    {
      v17 = v29;
      do
      {
        v7 = v26;
        v18 = *v17;
        v19 = v26;
        if ( v26 >= (unsigned __int64)v27 )
        {
          v20 = v18 & 0xFFFFFFFFFFFFFFFBLL;
          if ( v27 < (unsigned __int64)v26 + 1 )
          {
            v22 = v17;
            v23 = v16;
            sub_C8D5F0((__int64)&v25, v28, v26 + 1LL, 8u, v16, v9);
            v7 = v26;
            v17 = v22;
            v16 = v23;
          }
          v25[v7] = v20;
          ++v26;
        }
        else
        {
          v7 = (unsigned __int64)&v25[v26];
          if ( v7 )
          {
            *(_QWORD *)v7 = v18 & 0xFFFFFFFFFFFFFFFBLL;
            v19 = v26;
          }
          v26 = v19 + 1;
        }
        ++v17;
      }
      while ( (_QWORD *)v16 != v17 );
      v10 = v29;
    }
LABEL_8:
    if ( v10 != v31 )
      break;
LABEL_2:
    v4 = v26;
    if ( !v26 )
      goto LABEL_10;
LABEL_3:
    v3 = v25;
  }
  _libc_free(v10, v7);
  v4 = v26;
  if ( v26 )
    goto LABEL_3;
LABEL_10:
  v11 = sub_D98300((__int64)a1, a2);
  if ( v25 != v28 )
    _libc_free(v25, a2);
  return v11;
}
