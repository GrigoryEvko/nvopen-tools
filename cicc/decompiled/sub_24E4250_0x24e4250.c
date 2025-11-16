// Function: sub_24E4250
// Address: 0x24e4250
//
_QWORD *__fastcall sub_24E4250(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  _QWORD *v7; // rbx
  _QWORD *v8; // r15
  _QWORD *i; // r13
  unsigned int v10; // ebx
  void *v11; // rax
  unsigned int v12; // ebx
  void *v13; // rdi
  _BYTE *v14; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r11
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  size_t v26; // rdx
  size_t v27; // rdx
  __int64 v28; // [rsp+18h] [rbp-D8h]
  __int64 v29; // [rsp+20h] [rbp-D0h]
  void *v30; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+38h] [rbp-B8h]
  _BYTE v32[48]; // [rsp+40h] [rbp-B0h] BYREF
  void *src; // [rsp+70h] [rbp-80h] BYREF
  __int64 v34; // [rsp+78h] [rbp-78h]
  _BYTE v35[112]; // [rsp+80h] [rbp-70h] BYREF

  v7 = (_QWORD *)(a2 + 72);
  src = v35;
  v8 = *(_QWORD **)(a2 + 80);
  v34 = 0x800000000LL;
  v30 = v32;
  v31 = 0x600000000LL;
  if ( (_QWORD *)(a2 + 72) == v8 )
  {
    i = 0;
  }
  else
  {
    if ( !v8 )
      BUG();
    while ( 1 )
    {
      i = (_QWORD *)v8[4];
      if ( i != v8 + 3 )
        break;
      v8 = (_QWORD *)v8[1];
      if ( v7 == v8 )
        goto LABEL_7;
      if ( !v8 )
        BUG();
    }
  }
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      if ( !i )
        BUG();
      v16 = i[5];
      if ( v16 )
      {
        v17 = sub_B14240(v16);
        v19 = v18;
        if ( v18 != v17 )
        {
          while ( *(_BYTE *)(v17 + 32) )
          {
            v17 = *(_QWORD *)(v17 + 8);
            if ( v17 == v18 )
              goto LABEL_28;
          }
          if ( v17 != v18 )
          {
            v20 = (unsigned int)v31;
            a5 = (unsigned int)v31 + 1LL;
            if ( a5 > HIDWORD(v31) )
            {
LABEL_38:
              v28 = v19;
              v29 = v17;
              sub_C8D5F0((__int64)&v30, v32, a5, 8u, a5, a6);
              v20 = (unsigned int)v31;
              v19 = v28;
              v17 = v29;
            }
LABEL_25:
            *((_QWORD *)v30 + v20) = v17;
            v20 = (unsigned int)(v31 + 1);
            LODWORD(v31) = v31 + 1;
            while ( 1 )
            {
              v17 = *(_QWORD *)(v17 + 8);
              if ( v17 == v19 )
                break;
              if ( !*(_BYTE *)(v17 + 32) )
              {
                if ( v17 == v19 )
                  break;
                a5 = v20 + 1;
                if ( v20 + 1 <= (unsigned __int64)HIDWORD(v31) )
                  goto LABEL_25;
                goto LABEL_38;
              }
            }
          }
        }
      }
LABEL_28:
      if ( *((_BYTE *)i - 24) != 85 )
        goto LABEL_29;
      v22 = *(i - 7);
      if ( !v22 || *(_BYTE *)v22 || *(_QWORD *)(v22 + 24) != i[7] || (*(_BYTE *)(v22 + 33) & 0x20) == 0 )
        goto LABEL_29;
      v23 = *(_DWORD *)(v22 + 36);
      if ( v23 <= 0x45 )
        break;
      if ( v23 == 71 )
      {
        v24 = (unsigned int)v34;
        v25 = (unsigned int)v34 + 1LL;
        if ( v25 > HIDWORD(v34) )
          goto LABEL_56;
        goto LABEL_48;
      }
LABEL_29:
      for ( i = (_QWORD *)i[1]; ; i = (_QWORD *)v8[4] )
      {
        v21 = v8 - 3;
        if ( !v8 )
          v21 = 0;
        if ( i != v21 + 6 )
          break;
        v8 = (_QWORD *)v8[1];
        if ( v7 == v8 )
          goto LABEL_7;
        if ( !v8 )
          BUG();
      }
      if ( v8 == v7 )
        goto LABEL_7;
    }
    if ( v23 <= 0x43 )
      goto LABEL_29;
    v24 = (unsigned int)v34;
    v25 = (unsigned int)v34 + 1LL;
    if ( v25 > HIDWORD(v34) )
    {
LABEL_56:
      sub_C8D5F0((__int64)&src, v35, v25, 8u, a5, a6);
      v24 = (unsigned int)v34;
    }
LABEL_48:
    *((_QWORD *)src + v24) = i - 3;
    LODWORD(v34) = v34 + 1;
    goto LABEL_29;
  }
LABEL_7:
  v10 = v34;
  v11 = a1 + 2;
  *a1 = a1 + 2;
  a1[1] = 0x800000000LL;
  if ( v10 )
  {
    v26 = 8LL * v10;
    if ( v10 <= 8
      || (sub_C8D5F0((__int64)a1, a1 + 2, v10, 8u, v10, a6), v11 = (void *)*a1, (v26 = 8LL * (unsigned int)v34) != 0) )
    {
      memcpy(v11, src, v26);
    }
    *((_DWORD *)a1 + 2) = v10;
  }
  v12 = v31;
  v13 = a1 + 12;
  a1[10] = a1 + 12;
  a1[11] = 0x600000000LL;
  if ( !v12 )
  {
    v14 = v30;
    goto LABEL_10;
  }
  if ( v12 > 6 )
  {
    sub_C8D5F0((__int64)(a1 + 10), a1 + 12, v12, 8u, (__int64)(a1 + 10), a6);
    v13 = (void *)a1[10];
    v14 = v30;
    v27 = 8LL * (unsigned int)v31;
    if ( !v27 )
      goto LABEL_60;
  }
  else
  {
    v14 = v30;
    v27 = 8LL * v12;
  }
  memcpy(v13, v14, v27);
  v14 = v30;
LABEL_60:
  *((_DWORD *)a1 + 22) = v12;
LABEL_10:
  if ( v14 != v32 )
    _libc_free((unsigned __int64)v14);
  if ( src != v35 )
    _libc_free((unsigned __int64)src);
  return a1;
}
