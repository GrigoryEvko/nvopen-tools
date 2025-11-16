// Function: sub_15BA070
// Address: 0x15ba070
//
__int64 __fastcall sub_15BA070(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // r13
  _BYTE *v5; // rsi
  bool v6; // zf
  __int64 v7; // rbx
  _BYTE *v8; // r8
  __int64 v9; // rbx
  unsigned int v10; // edx
  unsigned __int64 v11; // rcx
  _QWORD *v12; // rax
  _QWORD *v13; // r14
  __int64 v14; // r8
  __int64 result; // rax
  __int64 *v16; // rdi
  _QWORD *v17; // rdi
  _QWORD *v18; // rax
  _QWORD *v19; // rcx
  unsigned __int64 v20; // rdx
  __int64 v21; // [rsp-A0h] [rbp-A0h]
  __int64 v22; // [rsp-98h] [rbp-98h] BYREF
  _BYTE *v23; // [rsp-90h] [rbp-90h]
  _BYTE *v24; // [rsp-88h] [rbp-88h]
  __int64 v25; // [rsp-80h] [rbp-80h]
  int v26; // [rsp-78h] [rbp-78h]
  _BYTE v27[112]; // [rsp-70h] [rbp-70h] BYREF

  if ( !a1 )
    return 0;
  v3 = a2;
  if ( !a2 )
    return 0;
  if ( a1 == a2 || !sub_15AF920(a1, a2) )
    return a1;
  if ( !a3 )
    return 0;
  v5 = v27;
  v6 = *(_DWORD *)(a1 + 8) == 2;
  v22 = 0;
  v23 = v27;
  v24 = v27;
  v25 = 8;
  v26 = 0;
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 - 8);
    if ( v7 )
    {
      v8 = v27;
      while ( 1 )
      {
        if ( v5 != v8 )
          goto LABEL_10;
        v17 = &v5[8 * HIDWORD(v25)];
        if ( v17 != (_QWORD *)v5 )
        {
          v18 = v5;
          v19 = 0;
          while ( *v18 != v7 )
          {
            if ( *v18 == -2 )
              v19 = v18;
            if ( v17 == ++v18 )
            {
              if ( !v19 )
                goto LABEL_54;
              *v19 = v7;
              v8 = v24;
              --v26;
              v5 = v23;
              ++v22;
              goto LABEL_11;
            }
          }
          goto LABEL_11;
        }
LABEL_54:
        if ( HIDWORD(v25) < (unsigned int)v25 )
        {
          ++HIDWORD(v25);
          *v17 = v7;
          v5 = v23;
          ++v22;
          v8 = v24;
        }
        else
        {
LABEL_10:
          sub_16CCBA0(&v22, v7);
          v8 = v24;
          v5 = v23;
        }
LABEL_11:
        if ( *(_DWORD *)(v7 + 8) == 2 )
        {
          v7 = *(_QWORD *)(v7 - 8);
          if ( v7 )
            continue;
        }
        break;
      }
    }
  }
  v9 = *(unsigned int *)(v3 + 8);
  v10 = v9;
  if ( (_DWORD)v9 != 2 )
    goto LABEL_24;
  if ( !*(_QWORD *)(v3 - 8) )
  {
LABEL_25:
    v14 = 0;
    goto LABEL_26;
  }
  v3 = *(_QWORD *)(v3 - 8);
  v11 = (unsigned __int64)v24;
  v12 = v23;
  if ( v24 == v23 )
    goto LABEL_34;
LABEL_16:
  v13 = (_QWORD *)(v11 + 8LL * (unsigned int)v25);
  v12 = (_QWORD *)sub_16CC9F0(&v22, v3);
  if ( *v12 == v3 )
  {
    v11 = (unsigned __int64)v24;
    v20 = (unsigned __int64)(v24 == v23 ? &v24[8 * HIDWORD(v25)] : &v24[8 * (unsigned int)v25]);
  }
  else
  {
    v11 = (unsigned __int64)v24;
    if ( v24 != v23 )
    {
      v12 = &v24[8 * (unsigned int)v25];
      goto LABEL_19;
    }
    v12 = &v24[8 * HIDWORD(v25)];
    v20 = (unsigned __int64)v12;
  }
  while ( 1 )
  {
    while ( (_QWORD *)v20 != v12 && *v12 >= 0xFFFFFFFFFFFFFFFELL )
      ++v12;
LABEL_19:
    v10 = *(_DWORD *)(v3 + 8);
    if ( v13 != v12 )
      break;
    if ( v10 != 2 )
      goto LABEL_24;
    if ( !*(_QWORD *)(v3 - 8) )
      goto LABEL_25;
    v3 = *(_QWORD *)(v3 - 8);
    v12 = v23;
    if ( (_BYTE *)v11 != v23 )
      goto LABEL_16;
LABEL_34:
    v13 = (_QWORD *)(v11 + 8LL * HIDWORD(v25));
    if ( (_QWORD *)v11 == v13 )
    {
      v20 = v11;
    }
    else
    {
      do
      {
        if ( *v12 == v3 )
          break;
        ++v12;
      }
      while ( v13 != v12 );
      v20 = v11 + 8LL * HIDWORD(v25);
    }
  }
  if ( v10 != 2 )
  {
LABEL_24:
    v9 = v10;
    goto LABEL_25;
  }
  v14 = *(_QWORD *)(v3 - 8);
LABEL_26:
  v16 = (__int64 *)(*(_QWORD *)(v3 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(v3 + 16) & 4) != 0 )
    v16 = (__int64 *)*v16;
  result = sub_15B9E00(v16, 0, 0, *(_QWORD *)(v3 - 8 * v9), v14, 0, 1);
  if ( v24 != v23 )
  {
    v21 = result;
    _libc_free((unsigned __int64)v24);
    return v21;
  }
  return result;
}
