// Function: sub_98BCC0
// Address: 0x98bcc0
//
__int64 __fastcall sub_98BCC0(__int64 a1, __int64 a2)
{
  int v3; // eax
  unsigned __int8 *v4; // rsi
  unsigned __int8 **v5; // r9
  unsigned __int8 **v6; // rbx
  unsigned __int8 **v7; // r15
  unsigned __int8 *v8; // r14
  unsigned __int8 **v9; // rax
  unsigned __int8 **v10; // rdx
  _QWORD *v11; // rdi
  char v12; // dl
  int v13; // eax
  __int64 result; // rax
  __int64 v15; // rax
  unsigned __int8 *v16; // rdx
  __int64 v17; // r11
  int v18; // eax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  int v21; // eax
  unsigned __int8 *v22; // rcx
  unsigned __int8 v23; // al
  __int64 *v24; // r11
  unsigned __int8 **v25; // rdi
  __int64 v26; // [rsp+8h] [rbp-158h]
  unsigned __int8 *v27; // [rsp+20h] [rbp-140h]
  unsigned __int8 v28; // [rsp+28h] [rbp-138h]
  unsigned __int8 v29; // [rsp+28h] [rbp-138h]
  _QWORD *v30; // [rsp+30h] [rbp-130h] BYREF
  unsigned int v31; // [rsp+38h] [rbp-128h]
  unsigned int v32; // [rsp+3Ch] [rbp-124h]
  _QWORD v33[4]; // [rsp+40h] [rbp-120h] BYREF
  unsigned __int8 **v34; // [rsp+60h] [rbp-100h] BYREF
  __int64 v35; // [rsp+68h] [rbp-F8h]
  _BYTE v36[32]; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v37; // [rsp+90h] [rbp-D0h] BYREF
  unsigned __int8 **v38; // [rsp+98h] [rbp-C8h]
  __int64 v39; // [rsp+A0h] [rbp-C0h]
  int v40; // [rsp+A8h] [rbp-B8h]
  char v41; // [rsp+ACh] [rbp-B4h]
  char v42; // [rsp+B0h] [rbp-B0h] BYREF

  v27 = (unsigned __int8 *)(a2 + 16);
  v37 = 0;
  v39 = 16;
  v40 = 0;
  v41 = 1;
  v32 = 4;
  v33[0] = a1;
  v38 = (unsigned __int8 **)&v42;
  v30 = v33;
  v3 = 1;
  while ( 1 )
  {
    v4 = (unsigned __int8 *)&v34;
    v31 = v3 - 1;
    v35 = 0x400000000LL;
    v34 = (unsigned __int8 **)v36;
    sub_98B4D0(a1, (__int64)&v34, 0, 6u);
    v5 = v34;
    v6 = &v34[(unsigned int)v35];
    if ( v6 != v34 )
      break;
LABEL_11:
    if ( v5 != (unsigned __int8 **)v36 )
      _libc_free(v5, v4);
    v3 = v31;
    v11 = v30;
    if ( !v31 )
    {
      result = 1;
      goto LABEL_56;
    }
    a1 = v30[v31 - 1];
  }
  v7 = v34;
  while ( 2 )
  {
    while ( 2 )
    {
      while ( 2 )
      {
        v8 = *v7;
        if ( v41 )
        {
          v9 = v38;
          v10 = &v38[HIDWORD(v39)];
          if ( v38 != v10 )
          {
            while ( v8 != *v9 )
            {
              if ( v10 == ++v9 )
                goto LABEL_23;
            }
            goto LABEL_9;
          }
LABEL_23:
          if ( HIDWORD(v39) < (unsigned int)v39 )
          {
            ++HIDWORD(v39);
            *v10 = v8;
            ++v37;
            v13 = *v8;
            if ( (unsigned __int8)v13 <= 0x1Cu )
              goto LABEL_25;
            goto LABEL_17;
          }
        }
        v4 = *v7;
        sub_C8CC70(&v37, *v7);
        if ( !v12 )
        {
LABEL_9:
          if ( v6 == ++v7 )
            goto LABEL_10;
          continue;
        }
        break;
      }
      v13 = *v8;
      if ( (unsigned __int8)v13 <= 0x1Cu )
      {
LABEL_25:
        if ( (_BYTE)v13 != 5 || *((_WORD *)v8 + 1) != 48 )
          break;
        goto LABEL_27;
      }
LABEL_17:
      if ( v13 != 77 )
        break;
LABEL_27:
      if ( (v8[7] & 0x40) != 0 )
        v16 = (unsigned __int8 *)*((_QWORD *)v8 - 1);
      else
        v16 = &v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
      while ( 1 )
      {
        v17 = *(_QWORD *)v16;
        v18 = **(unsigned __int8 **)v16;
        if ( (unsigned __int8)v18 <= 0x1Cu )
          break;
        v21 = v18 - 29;
        if ( v21 == 47 )
          goto LABEL_48;
LABEL_37:
        if ( v21 != 13 )
          goto LABEL_31;
        if ( (*(_BYTE *)(v17 + 7) & 0x40) != 0 )
          v16 = *(unsigned __int8 **)(v17 - 8);
        else
          v16 = (unsigned __int8 *)(v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF));
        v22 = (unsigned __int8 *)*((_QWORD *)v16 + 4);
        v23 = *v22;
        if ( *v22 != 17 )
        {
          if ( v23 <= 0x1Cu )
          {
            if ( v23 != 5 || *((_WORD *)v22 + 1) != 17 )
              goto LABEL_31;
          }
          else if ( v23 != 46 && v23 != 84 )
          {
            goto LABEL_31;
          }
        }
      }
      if ( (_BYTE)v18 != 5 )
        goto LABEL_31;
      v21 = *(unsigned __int16 *)(v17 + 2);
      if ( v21 != 47 )
        goto LABEL_37;
LABEL_48:
      if ( (*(_BYTE *)(v17 + 7) & 0x40) != 0 )
        v24 = *(__int64 **)(v17 - 8);
      else
        v24 = (__int64 *)(v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF));
      v17 = *v24;
LABEL_31:
      if ( *(_BYTE *)(*(_QWORD *)(v17 + 8) + 8LL) == 14 )
      {
        v19 = v31;
        v20 = v31 + 1LL;
        if ( v20 > v32 )
        {
          v4 = (unsigned __int8 *)v33;
          v26 = v17;
          sub_C8D5F0(&v30, v33, v20, 8);
          v19 = v31;
          v17 = v26;
        }
        ++v7;
        v30[v19] = v17;
        ++v31;
        if ( v6 == v7 )
          goto LABEL_10;
        continue;
      }
      break;
    }
    result = sub_CF7060(v8);
    if ( (_BYTE)result )
    {
      v15 = *(unsigned int *)(a2 + 8);
      if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v4 = v27;
        sub_C8D5F0(a2, v27, v15 + 1, 8);
        v15 = *(unsigned int *)(a2 + 8);
      }
      ++v7;
      *(_QWORD *)(*(_QWORD *)a2 + 8 * v15) = v8;
      ++*(_DWORD *)(a2 + 8);
      if ( v6 == v7 )
      {
LABEL_10:
        v5 = v34;
        goto LABEL_11;
      }
      continue;
    }
    break;
  }
  v25 = v34;
  *(_DWORD *)(a2 + 8) = 0;
  if ( v25 == (unsigned __int8 **)v36 )
  {
    v11 = v30;
  }
  else
  {
    _libc_free(v25, v4);
    v11 = v30;
    result = 0;
  }
LABEL_56:
  if ( v11 != v33 )
  {
    v28 = result;
    _libc_free(v11, v4);
    result = v28;
  }
  if ( !v41 )
  {
    v29 = result;
    _libc_free(v38, v4);
    return v29;
  }
  return result;
}
