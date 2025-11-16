// Function: sub_2753060
// Address: 0x2753060
//
__int64 __fastcall sub_2753060(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 v4; // rbx
  __int64 v5; // rdi
  unsigned int v6; // r13d
  __int64 i; // rcx
  __int64 v8; // rax
  __int64 v9; // rax
  unsigned __int8 *v10; // r15
  unsigned int v11; // eax
  unsigned __int8 **v12; // rcx
  unsigned __int8 *v13; // rdi
  unsigned __int8 *v15; // rdi
  unsigned __int8 **v16; // rax
  __int64 v17; // rdx
  unsigned __int8 **v18; // rsi
  __int64 v19; // r10
  __int64 v20; // rdx
  unsigned __int8 **v21; // rdx
  unsigned int v22; // eax
  unsigned __int8 *v23; // r10
  int v24; // r11d
  char v25; // al
  int v26; // eax
  int v27; // ecx
  int v28; // r8d
  __int64 v29; // [rsp+10h] [rbp-F0h]
  __int64 v30; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+28h] [rbp-D8h]
  __int64 v32; // [rsp+30h] [rbp-D0h]
  __int64 v33; // [rsp+38h] [rbp-C8h]
  unsigned __int8 **v34; // [rsp+40h] [rbp-C0h]
  __int64 v35; // [rsp+48h] [rbp-B8h]
  _BYTE v36[176]; // [rsp+50h] [rbp-B0h] BYREF

  v2 = a1 + 72;
  v4 = *(_QWORD *)(a1 + 80);
  v34 = (unsigned __int8 **)v36;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v35 = 0x1000000000LL;
  if ( a1 + 72 == v4 )
  {
    v5 = 0;
  }
  else
  {
    if ( !v4 )
      BUG();
    while ( 1 )
    {
      v5 = *(_QWORD *)(v4 + 32);
      if ( v5 != v4 + 24 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v2 == v4 )
        break;
      if ( !v4 )
        BUG();
    }
  }
  v6 = 0;
  while ( v4 != v2 )
  {
    for ( i = *(_QWORD *)(v5 + 8); ; i = *(_QWORD *)(v4 + 32) )
    {
      v8 = v4 - 24;
      if ( !v4 )
        v8 = 0;
      if ( i != v8 + 48 )
        break;
      v4 = *(_QWORD *)(v4 + 8);
      if ( v2 == v4 )
        break;
      if ( !v4 )
        BUG();
    }
    v15 = (unsigned __int8 *)(v5 - 24);
    if ( (_DWORD)v32 )
    {
      if ( (_DWORD)v33 )
      {
        v22 = (v33 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v23 = *(unsigned __int8 **)(v31 + 8LL * v22);
        if ( v15 == v23 )
          goto LABEL_35;
        v24 = 1;
        while ( v23 != (unsigned __int8 *)-4096LL )
        {
          v22 = (v33 - 1) & (v24 + v22);
          v23 = *(unsigned __int8 **)(v31 + 8LL * v22);
          if ( v15 == v23 )
            goto LABEL_35;
          ++v24;
        }
      }
      goto LABEL_40;
    }
    v16 = v34;
    v17 = 8LL * (unsigned int)v35;
    v18 = &v34[(unsigned __int64)v17 / 8];
    v19 = v17 >> 3;
    v20 = v17 >> 5;
    if ( v20 )
    {
      v21 = &v34[4 * v20];
      while ( v15 != *v16 )
      {
        if ( v15 == v16[1] )
        {
          ++v16;
          break;
        }
        if ( v15 == v16[2] )
        {
          v16 += 2;
          break;
        }
        if ( v15 == v16[3] )
        {
          v16 += 3;
          break;
        }
        v16 += 4;
        if ( v21 == v16 )
        {
          v19 = v18 - v16;
          goto LABEL_47;
        }
      }
LABEL_34:
      if ( v18 != v16 )
        goto LABEL_35;
      goto LABEL_40;
    }
LABEL_47:
    if ( v19 != 2 )
    {
      if ( v19 != 3 )
      {
        if ( v19 == 1 && v15 == *v16 )
          goto LABEL_34;
        goto LABEL_40;
      }
      if ( v15 == *v16 )
        goto LABEL_34;
      ++v16;
    }
    if ( v15 == *v16 )
      goto LABEL_34;
    if ( v15 == *++v16 )
      goto LABEL_34;
LABEL_40:
    v29 = i;
    v25 = sub_F50EE0(v15, a2);
    i = v29;
    if ( v25 )
    {
      v26 = sub_2752980((__int64)v15, (__int64)&v30, a2);
      i = v29;
      v6 |= v26;
    }
LABEL_35:
    v5 = i;
  }
  while ( 1 )
  {
    v9 = (unsigned int)v35;
    if ( !(_DWORD)v35 )
      break;
    while ( 1 )
    {
      v10 = v34[v9 - 1];
      if ( (_DWORD)v33 )
      {
        v11 = (v33 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v12 = (unsigned __int8 **)(v31 + 8LL * v11);
        v13 = *v12;
        if ( *v12 == v10 )
        {
LABEL_20:
          *v12 = (unsigned __int8 *)-8192LL;
          LODWORD(v32) = v32 - 1;
          ++HIDWORD(v32);
        }
        else
        {
          v27 = 1;
          while ( v13 != (unsigned __int8 *)-4096LL )
          {
            v28 = v27 + 1;
            v11 = (v33 - 1) & (v27 + v11);
            v12 = (unsigned __int8 **)(v31 + 8LL * v11);
            v13 = *v12;
            if ( v10 == *v12 )
              goto LABEL_20;
            v27 = v28;
          }
        }
      }
      LODWORD(v35) = v35 - 1;
      if ( sub_F50EE0(v10, a2) )
        break;
      v9 = (unsigned int)v35;
      if ( !(_DWORD)v35 )
        goto LABEL_23;
    }
    v6 |= sub_2752980((__int64)v10, (__int64)&v30, a2);
  }
LABEL_23:
  if ( v34 != (unsigned __int8 **)v36 )
    _libc_free((unsigned __int64)v34);
  sub_C7D6A0(v31, 8LL * (unsigned int)v33, 8);
  return v6;
}
