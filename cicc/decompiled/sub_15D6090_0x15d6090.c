// Function: sub_15D6090
// Address: 0x15d6090
//
void __fastcall sub_15D6090(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r15
  int v7; // r15d
  int v8; // r15d
  int v9; // r15d
  int v10; // r15d
  __int64 v11; // rax
  __int64 *v12; // r8
  __int64 v13; // r10
  __int64 *v14; // rbx
  __int64 v15; // r9
  char *i; // rdi
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  char *v21; // rax
  char *v22; // rsi
  __int64 v23; // rcx
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  char *v27; // rax
  __int64 v28; // rcx
  unsigned __int64 v29; // rdi
  __int64 *v30; // [rsp+8h] [rbp-98h]
  __int64 *v32; // [rsp+18h] [rbp-88h]
  __int64 *v33; // [rsp+20h] [rbp-80h] BYREF
  int v34; // [rsp+28h] [rbp-78h]
  _BYTE v35[112]; // [rsp+30h] [rbp-70h] BYREF

  v3 = *(__int64 **)a1;
  v4 = 8LL * *(unsigned int *)(a1 + 8);
  v30 = (__int64 *)(*(_QWORD *)a1 + v4);
  v5 = v4 >> 3;
  v6 = v4 >> 5;
  if ( v6 )
  {
    v32 = &v3[4 * v6];
    while ( 1 )
    {
      sub_15CF8B0((__int64)&v33, *v3, a2);
      v10 = v34;
      if ( v33 != (__int64 *)v35 )
        _libc_free((unsigned __int64)v33);
      if ( v10 )
        goto LABEL_16;
      sub_15CF8B0((__int64)&v33, v3[1], a2);
      v7 = v34;
      if ( v33 != (__int64 *)v35 )
        _libc_free((unsigned __int64)v33);
      if ( v7 )
      {
        ++v3;
        goto LABEL_16;
      }
      sub_15CF8B0((__int64)&v33, v3[2], a2);
      v8 = v34;
      if ( v33 != (__int64 *)v35 )
        _libc_free((unsigned __int64)v33);
      if ( v8 )
      {
        v3 += 2;
        goto LABEL_16;
      }
      sub_15CF8B0((__int64)&v33, v3[3], a2);
      v9 = v34;
      if ( v33 != (__int64 *)v35 )
        _libc_free((unsigned __int64)v33);
      if ( v9 )
        break;
      v3 += 4;
      if ( v3 == v32 )
      {
        v5 = v30 - v3;
        goto LABEL_55;
      }
    }
    v3 += 3;
    goto LABEL_16;
  }
LABEL_55:
  if ( v5 == 2 )
    goto LABEL_62;
  if ( v5 == 3 )
  {
    if ( (unsigned __int8)sub_15CFAA0(*v3, a2) )
      goto LABEL_16;
    ++v3;
LABEL_62:
    if ( (unsigned __int8)sub_15CFAA0(*v3, a2) )
      goto LABEL_16;
    ++v3;
    goto LABEL_58;
  }
  if ( v5 != 1 )
    return;
LABEL_58:
  if ( !(unsigned __int8)sub_15CFAA0(*v3, a2) )
    return;
LABEL_16:
  if ( v30 == v3 )
    return;
  sub_15D57B0(&v33, a1, a2);
  v11 = *(unsigned int *)(a1 + 8);
  if ( v34 != (_DWORD)v11 )
  {
LABEL_49:
    sub_15D5DA0(a1, a2);
    v29 = (unsigned __int64)v33;
    if ( v33 == (__int64 *)v35 )
      return;
    goto LABEL_50;
  }
  v12 = *(__int64 **)a1;
  v13 = *(_QWORD *)a1 + 8 * v11;
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = v33;
    while ( *v12 == *v14 )
    {
      ++v12;
      ++v14;
      if ( (__int64 *)v13 == v12 )
        goto LABEL_52;
    }
    if ( (__int64 *)v13 != v12 )
    {
      v15 = *v12;
      for ( i = (char *)v12; ; v15 = *(_QWORD *)i )
      {
        v17 = (i - (char *)v12) >> 5;
        v18 = (i - (char *)v12) >> 3;
        if ( v17 > 0 )
        {
          v19 = *(_QWORD *)i;
          v20 = *v12;
          v21 = (char *)v12;
          v22 = (char *)&v12[4 * v17];
          while ( v20 != v19 )
          {
            if ( v19 == *((_QWORD *)v21 + 1) )
            {
              v21 += 8;
              break;
            }
            if ( v19 == *((_QWORD *)v21 + 2) )
            {
              v21 += 16;
              break;
            }
            if ( v19 == *((_QWORD *)v21 + 3) )
            {
              v21 += 24;
              break;
            }
            v21 += 32;
            if ( v22 == v21 )
            {
              v18 = (i - v21) >> 3;
              goto LABEL_36;
            }
            v20 = *(_QWORD *)v21;
          }
LABEL_32:
          if ( v21 != i )
            goto LABEL_33;
          goto LABEL_40;
        }
        v21 = (char *)v12;
LABEL_36:
        if ( v18 != 2 )
        {
          if ( v18 != 3 )
          {
            if ( v18 != 1 )
              goto LABEL_40;
            goto LABEL_39;
          }
          if ( *(_QWORD *)v21 == *(_QWORD *)i )
            goto LABEL_32;
          v21 += 8;
        }
        if ( *(_QWORD *)v21 == *(_QWORD *)i )
          goto LABEL_32;
        v21 += 8;
LABEL_39:
        if ( *(_QWORD *)v21 == *(_QWORD *)i )
          goto LABEL_32;
LABEL_40:
        v23 = *v14;
        v24 = v14;
        v25 = 0;
        while ( 1 )
        {
          ++v24;
          v25 += v23 == v15;
          if ( v24 == (__int64 *)((char *)v14 + v13 - (_QWORD)v12) )
            break;
          v23 = *v24;
        }
        if ( !v25 )
          goto LABEL_49;
        if ( (char *)v13 == i )
          goto LABEL_49;
        v26 = v15;
        v27 = i;
        v28 = 0;
        while ( 1 )
        {
          v27 += 8;
          v28 += v26 == v15;
          if ( (char *)v13 == v27 )
            break;
          v26 = *(_QWORD *)v27;
        }
        if ( v25 != v28 )
          goto LABEL_49;
LABEL_33:
        i += 8;
        if ( (char *)v13 == i )
          break;
      }
    }
  }
LABEL_52:
  v29 = (unsigned __int64)v33;
  if ( v33 != (__int64 *)v35 )
LABEL_50:
    _libc_free(v29);
}
