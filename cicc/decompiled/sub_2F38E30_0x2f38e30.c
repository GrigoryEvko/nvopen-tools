// Function: sub_2F38E30
// Address: 0x2f38e30
//
unsigned __int64 __fastcall sub_2F38E30(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rax
  char v7; // r14
  unsigned __int64 v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rsi
  __int64 v13; // rdx
  char *v14; // rax
  char *v15; // rdx
  unsigned __int64 result; // rax
  __int64 v17; // rax
  char *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rax
  int v23; // eax
  char v24; // al
  unsigned __int64 v25; // rdx
  __int64 v26; // [rsp+8h] [rbp-98h]
  unsigned __int64 v27; // [rsp+8h] [rbp-98h]
  __int64 v28; // [rsp+10h] [rbp-90h] BYREF
  char *v29; // [rsp+18h] [rbp-88h]
  __int64 v30; // [rsp+20h] [rbp-80h]
  int v31; // [rsp+28h] [rbp-78h]
  char v32; // [rsp+2Ch] [rbp-74h]
  char v33; // [rsp+30h] [rbp-70h] BYREF

  v6 = a1[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v6 == a1 + 6 )
    return a1[7];
  v7 = *(_BYTE *)(a2 + 216);
  v8 = a1[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v7 && !*(_BYTE *)(a2 + 262) )
    return sub_2E313E0((__int64)a1);
  v30 = 8;
  v29 = &v33;
  v9 = a1[4];
  v28 = 0;
  v31 = 0;
  v10 = *(_QWORD *)(v9 + 32);
  v32 = 1;
  if ( (int)a3 < 0 )
  {
    a3 = *(_QWORD *)(v10 + 56) + 16 * (a3 & 0x7FFFFFFF);
    v11 = *(_QWORD *)(a3 + 8);
  }
  else
  {
    a3 = (unsigned int)a3;
    v11 = *(_QWORD *)(*(_QWORD *)(v10 + 304) + 8LL * (unsigned int)a3);
  }
  if ( v11 )
  {
    if ( (*(_BYTE *)(v11 + 3) & 0x10) != 0 || (v11 = *(_QWORD *)(v11 + 32)) != 0 && (*(_BYTE *)(v11 + 3) & 0x10) != 0 )
    {
      v12 = *(_QWORD *)(v11 + 16);
      if ( a1 == *(_QWORD **)(v12 + 24) )
        goto LABEL_28;
      while ( 1 )
      {
        while ( 1 )
        {
          v11 = *(_QWORD *)(v11 + 32);
          if ( !v11 || (*(_BYTE *)(v11 + 3) & 0x10) == 0 )
          {
            v6 = a1[6] & 0xFFFFFFFFFFFFFFF8LL;
            v8 = v6;
            goto LABEL_12;
          }
          v17 = *(_QWORD *)(v11 + 16);
          if ( v17 != v12 )
          {
            v12 = *(_QWORD *)(v11 + 16);
            if ( a1 == *(_QWORD **)(v17 + 24) )
              break;
          }
        }
LABEL_28:
        if ( v32 )
        {
          v18 = v29;
          a3 = (__int64)&v29[8 * HIDWORD(v30)];
          if ( v29 == (char *)a3 )
          {
LABEL_53:
            if ( HIDWORD(v30) >= (unsigned int)v30 )
              goto LABEL_54;
            ++HIDWORD(v30);
            *(_QWORD *)a3 = v12;
            v12 = *(_QWORD *)(v11 + 16);
            ++v28;
          }
          else
          {
            while ( *(_QWORD *)v18 != v12 )
            {
              v18 += 8;
              if ( (char *)a3 == v18 )
                goto LABEL_53;
            }
            v12 = *(_QWORD *)(v11 + 16);
          }
        }
        else
        {
LABEL_54:
          sub_C8CC70((__int64)&v28, v12, a3, a4, a5, a6);
          v12 = *(_QWORD *)(v11 + 16);
        }
      }
    }
  }
LABEL_12:
  v26 = a1[7];
  if ( !v6 )
    goto LABEL_67;
  v13 = *(_QWORD *)v6;
  if ( (*(_QWORD *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v25 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      v8 = v25;
      if ( (*(_BYTE *)(v25 + 44) & 4) == 0 )
        break;
      v13 = *(_QWORD *)v25;
    }
  }
  while ( 1 )
  {
    if ( a1 + 6 == (_QWORD *)v8 )
      goto LABEL_49;
    if ( v32 )
    {
      v14 = v29;
      v15 = &v29[8 * HIDWORD(v30)];
      if ( v29 == v15 )
        goto LABEL_35;
      while ( *(_QWORD *)v14 != v8 )
      {
        v14 += 8;
        if ( v15 == v14 )
          goto LABEL_35;
      }
LABEL_20:
      if ( v8 )
      {
        if ( (*(_BYTE *)v8 & 4) == 0 && (*(_BYTE *)(v8 + 44) & 8) != 0 )
        {
          do
            v8 = *(_QWORD *)(v8 + 8);
          while ( (*(_BYTE *)(v8 + 44) & 8) != 0 );
        }
        result = sub_2E31210((__int64)a1, *(_QWORD *)(v8 + 8));
        goto LABEL_50;
      }
LABEL_67:
      BUG();
    }
    if ( sub_C8CA60((__int64)&v28, v8) )
      goto LABEL_20;
LABEL_35:
    if ( v7 )
    {
      v23 = *(_DWORD *)(v8 + 44);
      if ( (v23 & 4) != 0 || (v23 & 8) == 0 )
        v24 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v8 + 16) + 24LL) >> 7;
      else
        v24 = sub_2E88A90(v8, 128, 1);
      if ( v24 )
        break;
    }
    if ( *(_WORD *)(v8 + 68) == 2 )
      break;
    v19 = (_QWORD *)(*(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL);
    v20 = v19;
    if ( !v19 )
      goto LABEL_67;
    v8 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
    v21 = *v19;
    if ( (v21 & 4) == 0 && (*((_BYTE *)v20 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        v22 = v21 & 0xFFFFFFFFFFFFFFF8LL;
        v8 = v22;
        if ( (*(_BYTE *)(v22 + 44) & 4) == 0 )
          break;
        v21 = *(_QWORD *)v22;
      }
    }
  }
  v26 = v8;
LABEL_49:
  result = sub_2E31210((__int64)a1, v26);
LABEL_50:
  if ( !v32 )
  {
    v27 = result;
    _libc_free((unsigned __int64)v29);
    return v27;
  }
  return result;
}
