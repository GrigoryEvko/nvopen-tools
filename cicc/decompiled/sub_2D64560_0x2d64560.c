// Function: sub_2D64560
// Address: 0x2d64560
//
__int64 __fastcall sub_2D64560(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // r11
  __int64 v8; // r10
  __int64 v9; // r8
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r15
  unsigned int v18; // eax
  __int64 v19; // rbx
  __int64 v20; // r15
  _QWORD *i; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // r15
  _QWORD *v24; // rax
  __int64 v25; // rdx
  char *v26; // rdx
  __int64 v27; // r15
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r9
  __int64 v33; // rsi
  __int64 v34; // rdi
  int v35; // edx
  __int64 v36; // rax
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rdx
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // [rsp+8h] [rbp-108h]
  int v47; // [rsp+8h] [rbp-108h]
  __int64 v48; // [rsp+10h] [rbp-100h]
  unsigned int v49; // [rsp+10h] [rbp-100h]
  unsigned __int8 v50; // [rsp+18h] [rbp-F8h]
  _QWORD *v51; // [rsp+20h] [rbp-F0h] BYREF
  unsigned __int64 v52; // [rsp+28h] [rbp-E8h]
  __int64 v53; // [rsp+40h] [rbp-D0h] BYREF
  char *v54; // [rsp+48h] [rbp-C8h]
  __int64 v55; // [rsp+50h] [rbp-C0h]
  int v56; // [rsp+58h] [rbp-B8h]
  char v57; // [rsp+5Ch] [rbp-B4h]
  char v58; // [rsp+60h] [rbp-B0h] BYREF

  v3 = a1;
  v4 = sub_AA5930(a1);
  v7 = v6;
  v8 = v4;
  while ( v7 != v8 )
  {
    v9 = *(_QWORD *)(v8 + 16);
    if ( v9 )
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v9 + 24);
        if ( *(_QWORD *)(v10 + 40) != a2 || *(_BYTE *)v10 != 84 )
          return 0;
        v5 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
        if ( (*(_DWORD *)(v10 + 4) & 0x7FFFFFF) != 0 )
          break;
LABEL_13:
        v9 = *(_QWORD *)(v9 + 8);
        if ( !v9 )
          goto LABEL_14;
      }
      v11 = *(_QWORD *)(v10 - 8);
      v12 = 0;
      v5 = 8LL * (unsigned int)v5;
      while ( 1 )
      {
        v13 = *(_QWORD *)(v11 + 4 * v12);
        if ( *(_BYTE *)v13 > 0x1Cu
          && v3 == *(_QWORD *)(v13 + 40)
          && v3 != *(_QWORD *)(v11 + 32LL * *(unsigned int *)(v10 + 72) + v12) )
        {
          return 0;
        }
        v12 += 8;
        if ( v5 == v12 )
          goto LABEL_13;
      }
    }
LABEL_14:
    v15 = *(_QWORD *)(v8 + 32);
    if ( !v15 )
      BUG();
    v8 = 0;
    if ( *(_BYTE *)(v15 - 24) == 84 )
      v8 = v15 - 24;
  }
  v16 = *(_QWORD *)(a2 + 56);
  if ( !v16 )
    BUG();
  if ( *(_BYTE *)(v16 - 24) != 84 )
    return 1;
  v17 = *(_QWORD *)(v3 + 56);
  v53 = 0;
  v54 = &v58;
  v55 = 16;
  v56 = 0;
  v57 = 1;
  if ( !v17 )
    BUG();
  if ( *(_BYTE *)(v17 - 24) != 84 )
  {
    v51 = *(_QWORD **)(v3 + 16);
    sub_2D63A10((__int64 *)&v51);
    v23 = (__int64)v51;
    if ( v51 )
    {
      v24 = sub_AE6EC0((__int64)&v53, *(_QWORD *)(v51[3] + 40LL));
LABEL_35:
      if ( v57 )
        v26 = &v54[8 * HIDWORD(v55)];
      else
        v26 = &v54[8 * (unsigned int)v55];
      v51 = v24;
      v52 = (unsigned __int64)v26;
      sub_254BBF0((__int64)&v51);
      while ( 1 )
      {
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          break;
        v25 = *(_QWORD *)(v23 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v25 - 30) <= 0xAu )
        {
          v24 = sub_AE6EC0((__int64)&v53, *(_QWORD *)(v25 + 40));
          goto LABEL_35;
        }
      }
    }
    goto LABEL_39;
  }
  v18 = *(_DWORD *)(v17 - 20) & 0x7FFFFFF;
  if ( v18 )
  {
    v46 = v3;
    v19 = v17;
    v20 = 0;
    v48 = 8LL * v18;
    for ( i = sub_AE6EC0((__int64)&v53, *(_QWORD *)(*(_QWORD *)(v19 - 32) + 32LL * *(unsigned int *)(v19 + 48)));
          ;
          i = sub_AE6EC0((__int64)&v53, *(_QWORD *)(*(_QWORD *)(v19 - 32) + 32LL * *(unsigned int *)(v19 + 48) + v20)) )
    {
      v22 = (unsigned __int64)(v57 ? &v54[8 * HIDWORD(v55)] : &v54[8 * (unsigned int)v55]);
      v51 = i;
      v20 += 8;
      v52 = v22;
      sub_254BBF0((__int64)&v51);
      if ( v48 == v20 )
        break;
    }
    v3 = v46;
LABEL_39:
    v47 = *(_DWORD *)(v16 - 20) & 0x7FFFFFF;
    if ( !v47 )
    {
LABEL_74:
      result = 1;
      goto LABEL_71;
    }
    goto LABEL_40;
  }
  v47 = *(_DWORD *)(v16 - 20) & 0x7FFFFFF;
  if ( !v47 )
    return 1;
LABEL_40:
  v27 = v3;
  v28 = v16;
  v49 = 0;
  while ( 1 )
  {
    v29 = *(_QWORD *)(*(_QWORD *)(v28 - 32) + 32LL * *(unsigned int *)(v28 + 48) + 8LL * v49);
    if ( (unsigned __int8)sub_B19060((__int64)&v53, v29, v49, v5) )
    {
      v30 = sub_AA5930(a2);
      v32 = v31;
      v33 = v30;
      if ( v30 != v31 )
        break;
    }
LABEL_41:
    if ( v47 == ++v49 )
      goto LABEL_74;
  }
  while ( 1 )
  {
    v34 = *(_QWORD *)(v33 - 8);
    v35 = *(_DWORD *)(v33 + 4) & 0x7FFFFFF;
    if ( !v35 )
    {
      v39 = *(_QWORD *)(v34 + 0x1FFFFFFFE0LL);
      if ( *(_BYTE *)v39 != 84 || v27 != *(_QWORD *)(v39 + 40) )
        goto LABEL_55;
      v37 = *(_QWORD *)(v34 + 0x1FFFFFFFE0LL);
LABEL_64:
      v41 = *(_QWORD *)(v39 - 8);
      v42 = 0x1FFFFFFFE0LL;
      v5 = *(_DWORD *)(v39 + 4) & 0x7FFFFFF;
      if ( (*(_DWORD *)(v39 + 4) & 0x7FFFFFF) != 0 )
      {
        v43 = *(unsigned int *)(v39 + 72);
        v44 = 0;
        v45 = v41 + 32 * v43;
        do
        {
          if ( v29 == *(_QWORD *)(v45 + 8 * v44) )
          {
            v42 = 32 * v44;
            goto LABEL_69;
          }
          ++v44;
        }
        while ( (_DWORD)v5 != (_DWORD)v44 );
        v42 = 0x1FFFFFFFE0LL;
      }
LABEL_69:
      v39 = *(_QWORD *)(v41 + v42);
      goto LABEL_54;
    }
    v36 = 0;
    v5 = v34 + 32LL * *(unsigned int *)(v33 + 72);
    do
    {
      if ( v29 == *(_QWORD *)(v5 + 8 * v36) )
      {
        v37 = *(_QWORD *)(v34 + 32 * v36);
        goto LABEL_49;
      }
      ++v36;
    }
    while ( v35 != (_DWORD)v36 );
    v37 = *(_QWORD *)(v34 + 0x1FFFFFFFE0LL);
LABEL_49:
    v38 = 0;
    do
    {
      if ( v27 == *(_QWORD *)(v5 + 8 * v38) )
      {
        v39 = *(_QWORD *)(v34 + 32 * v38);
        goto LABEL_53;
      }
      ++v38;
    }
    while ( v35 != (_DWORD)v38 );
    v39 = *(_QWORD *)(v34 + 0x1FFFFFFFE0LL);
LABEL_53:
    if ( *(_BYTE *)v39 == 84 && v27 == *(_QWORD *)(v39 + 40) )
      goto LABEL_64;
LABEL_54:
    if ( v39 != v37 )
      break;
LABEL_55:
    v40 = *(_QWORD *)(v33 + 32);
    if ( !v40 )
      BUG();
    v33 = 0;
    if ( *(_BYTE *)(v40 - 24) == 84 )
      v33 = v40 - 24;
    if ( v32 == v33 )
      goto LABEL_41;
  }
  result = 0;
LABEL_71:
  if ( !v57 )
  {
    v50 = result;
    _libc_free((unsigned __int64)v54);
    return v50;
  }
  return result;
}
