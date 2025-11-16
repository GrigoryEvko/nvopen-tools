// Function: sub_C50FA0
// Address: 0xc50fa0
//
__int64 __fastcall sub_C50FA0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, char a5)
{
  __int64 result; // rax
  _BYTE *v6; // r14
  _BYTE *v7; // rdx
  _BYTE *v8; // rbx
  __int64 v11; // r11
  unsigned __int64 v12; // r12
  _BYTE *v13; // r13
  __int64 v14; // r13
  _BYTE *v15; // rdi
  __int64 v16; // r15
  __int64 v17; // rbx
  char v18; // r15
  _BYTE *v19; // r9
  char v20; // bl
  _BYTE *v21; // r12
  __int64 v22; // rax
  __int64 v23; // rbx
  char v24; // r13
  char v26; // [rsp+Fh] [rbp-F1h]
  __int64 v27; // [rsp+18h] [rbp-E8h]
  char v28; // [rsp+18h] [rbp-E8h]
  char v29; // [rsp+18h] [rbp-E8h]
  char v30; // [rsp+18h] [rbp-E8h]
  char v31; // [rsp+18h] [rbp-E8h]
  __int64 v32; // [rsp+20h] [rbp-E0h]
  __int64 v33; // [rsp+20h] [rbp-E0h]
  __int64 v34; // [rsp+20h] [rbp-E0h]
  char v35; // [rsp+20h] [rbp-E0h]
  __int64 v36; // [rsp+20h] [rbp-E0h]
  __int64 v37; // [rsp+20h] [rbp-E0h]
  _BYTE *v38; // [rsp+30h] [rbp-D0h] BYREF
  _BYTE *v39; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v40; // [rsp+40h] [rbp-C0h]
  _BYTE v41[184]; // [rsp+48h] [rbp-B8h] BYREF

  result = (__int64)v41;
  v38 = v41;
  v39 = 0;
  v40 = 128;
  if ( !a2 )
    return result;
  result = (__int64)&v38;
  v6 = a2;
  v7 = 0;
  v8 = 0;
  v11 = 0x100002600LL;
  while ( 1 )
  {
    if ( !v7 )
    {
      v13 = v8;
      goto LABEL_18;
    }
    v12 = (unsigned __int8)v8[a1];
LABEL_5:
    v13 = v8 + 1;
    if ( (_BYTE)v12 == 92 && v13 < v6 )
    {
      v24 = v13[a1];
      if ( (unsigned __int64)(v7 + 1) > v40 )
      {
        a2 = v41;
        v31 = a5;
        v37 = a4;
        sub_C8D290(&v38, v41, v7 + 1, 1);
        v7 = v39;
        a5 = v31;
        v11 = 0x100002600LL;
        a4 = v37;
      }
      v8 += 2;
      v7[(_QWORD)v38] = v24;
      result = (__int64)v39;
      v7 = ++v39;
      goto LABEL_28;
    }
    if ( (_BYTE)v12 == 34 || (_BYTE)v12 == 39 )
    {
      if ( v13 == v6 )
      {
LABEL_29:
        v15 = v38;
        v16 = a4;
        if ( !v7 )
          goto LABEL_51;
LABEL_30:
        a2 = v15;
        v17 = sub_C948A0(a3, v15, v7);
        result = *(unsigned int *)(v16 + 8);
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(v16 + 12) )
        {
          a2 = (_BYTE *)(v16 + 16);
          sub_C8D5F0(v16, v16 + 16, result + 1, 8);
          result = *(unsigned int *)(v16 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v16 + 8 * result) = v17;
        ++*(_DWORD *)(v16 + 8);
        goto LABEL_33;
      }
      result = a1;
      v26 = a5;
      v18 = v12;
      while ( 2 )
      {
        v20 = v13[result];
        v21 = v13 + 1;
        if ( v20 == v18 )
        {
          a5 = v26;
          v8 = v13 + 1;
          a1 = result;
          goto LABEL_28;
        }
        if ( v20 == 92 )
        {
          if ( v6 == v21 )
          {
            v19 = v7 + 1;
            if ( v40 >= (unsigned __int64)(v7 + 1) )
            {
              v16 = a4;
              v7[(_QWORD)v38] = 92;
              result = (__int64)v39;
              v7 = ++v39;
              goto LABEL_50;
            }
LABEL_45:
            v13 = v21;
            v27 = a4;
            v32 = result;
            sub_C8D290(&v38, v41, v19, 1);
            v7 = v39;
            a4 = v27;
            v11 = 0x100002600LL;
            result = v32;
          }
          else
          {
            v20 = v21[result];
            v13 += 2;
            v19 = v7 + 1;
            if ( (unsigned __int64)(v7 + 1) > v40 )
            {
LABEL_43:
              v21 = v13;
              goto LABEL_45;
            }
          }
        }
        else
        {
          ++v13;
          v19 = v7 + 1;
          if ( (unsigned __int64)(v7 + 1) > v40 )
            goto LABEL_43;
        }
        v7[(_QWORD)v38] = v20;
        a2 = v39;
        v7 = ++v39;
        if ( v6 == v13 )
          goto LABEL_29;
        continue;
      }
    }
    if ( (unsigned __int8)v12 <= 0x20u && _bittest64(&v11, v12) )
      break;
    if ( (unsigned __int64)(v7 + 1) > v40 )
    {
      a2 = v41;
      v29 = a5;
      v34 = a4;
      sub_C8D290(&v38, v41, v7 + 1, 1);
      v7 = v39;
      a5 = v29;
      v11 = 0x100002600LL;
      a4 = v34;
    }
    ++v8;
    v7[(_QWORD)v38] = v12;
    result = (__int64)v39;
    v7 = ++v39;
LABEL_28:
    if ( v6 == v8 )
      goto LABEL_29;
  }
  if ( v7 )
  {
    a2 = v38;
    v28 = a5;
    v33 = a4;
    v22 = sub_C948A0(a3, v38, v7);
    a4 = v33;
    a5 = v28;
    v11 = 0x100002600LL;
    v23 = v22;
    result = *(unsigned int *)(v33 + 8);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(v33 + 12) )
    {
      a2 = (_BYTE *)(v33 + 16);
      sub_C8D5F0(v33, v33 + 16, result + 1, 8);
      a4 = v33;
      a5 = v28;
      v11 = 0x100002600LL;
      result = *(unsigned int *)(v33 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = v23;
    ++*(_DWORD *)(a4 + 8);
  }
  if ( (_BYTE)v12 == 10 && a5 )
  {
    result = *(unsigned int *)(a4 + 8);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
    {
      a2 = (_BYTE *)(a4 + 16);
      v30 = a5;
      v36 = a4;
      sub_C8D5F0(a4, a4 + 16, result + 1, 8);
      a4 = v36;
      a5 = v30;
      v11 = 0x100002600LL;
      result = *(unsigned int *)(v36 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a4 + 8 * result) = 0;
    ++*(_DWORD *)(a4 + 8);
  }
  v39 = 0;
  if ( v13 != v6 )
  {
LABEL_18:
    v8 = v13;
    v14 = a4;
    do
    {
      while ( 1 )
      {
        v12 = (unsigned __int8)v8[a1];
        if ( (unsigned __int8)v12 > 0x20u || !_bittest64(&v11, v12) )
        {
          v7 = v39;
          a4 = v14;
          goto LABEL_5;
        }
        if ( (_BYTE)v12 == 10 && a5 )
          break;
        if ( ++v8 == v6 )
          goto LABEL_49;
      }
      result = *(unsigned int *)(v14 + 8);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(v14 + 12) )
      {
        a2 = (_BYTE *)(v14 + 16);
        v35 = a5;
        sub_C8D5F0(v14, v14 + 16, result + 1, 8);
        result = *(unsigned int *)(v14 + 8);
        a5 = v35;
        v11 = 0x100002600LL;
      }
      ++v8;
      *(_QWORD *)(*(_QWORD *)v14 + 8 * result) = 0;
      ++*(_DWORD *)(v14 + 8);
    }
    while ( v8 != v6 );
LABEL_49:
    v7 = v39;
    v16 = v14;
LABEL_50:
    v15 = v38;
    if ( !v7 )
    {
LABEL_51:
      if ( v15 != v41 )
        return _libc_free(v15, a2);
      return result;
    }
    goto LABEL_30;
  }
LABEL_33:
  v15 = v38;
  if ( v38 != v41 )
    return _libc_free(v15, a2);
  return result;
}
