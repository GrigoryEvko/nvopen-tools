// Function: sub_E17AC0
// Address: 0xe17ac0
//
__int64 __fastcall sub_E17AC0(__int64 a1, _OWORD *a2)
{
  unsigned int v3; // r14d
  __int64 v4; // r13
  _OWORD *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  int v9; // eax
  char *v10; // r15
  char *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r8
  signed __int64 v14; // rdx
  char *v15; // rcx
  char *v17; // rax
  char *v18; // rax
  char *v19; // [rsp+10h] [rbp-90h]
  char *v20; // [rsp+18h] [rbp-88h]
  char *v21; // [rsp+20h] [rbp-80h]
  _OWORD src[4]; // [rsp+28h] [rbp-78h] BYREF
  char v23; // [rsp+68h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 16);
  v21 = &v23;
  v19 = (char *)src;
  v20 = (char *)src;
  memset(src, 0, sizeof(src));
  while ( 1 )
  {
    v5 = a2;
    v6 = (*(__int64 (__fastcall **)(__int64, _OWORD *))(*(_QWORD *)v4 + 24LL))(v4, a2);
    if ( *(_BYTE *)(v6 + 8) != 13 )
      break;
    v4 = *(_QWORD *)(v6 + 16);
    v9 = *(_DWORD *)(v6 + 24);
    v10 = v20;
    if ( (int)v3 > v9 )
      v3 = v9;
    if ( v20 == v21 )
    {
      if ( v19 == (char *)src )
      {
        v17 = (char *)malloc(16 * ((v20 - v19) >> 3), a2, v20 - v19, 16 * ((v20 - v19) >> 3), v7, v8);
        v13 = 16 * ((v20 - v19) >> 3);
        v14 = v20 - v19;
        v15 = v17;
        if ( !v17 )
LABEL_20:
          abort();
        if ( v20 != (char *)src )
        {
          v5 = src;
          v18 = (char *)memcpy(v17, src, v20 - v19);
          v13 = 16 * ((v20 - v19) >> 3);
          v14 = v20 - v19;
          v15 = v18;
        }
        v19 = v15;
      }
      else
      {
        v5 = (_OWORD *)(16 * ((v20 - v19) >> 3));
        v12 = realloc(v19);
        v13 = (__int64)v5;
        v14 = v20 - v19;
        v19 = (char *)v12;
        v15 = (char *)v12;
        if ( !v12 )
          goto LABEL_20;
      }
      v10 = &v15[v14];
      v21 = &v15[v13];
    }
    v20 = v10 + 8;
    *(_QWORD *)v10 = v4;
    v11 = v19;
    if ( (unsigned __int64)(v10 + 8 - v19) > 8
      && v4 == *(_QWORD *)&v19[8 * ((unsigned __int64)(((v10 + 8 - v19) >> 3) - 1) >> 1)] )
    {
      goto LABEL_13;
    }
  }
  v11 = v19;
LABEL_13:
  if ( v11 != (char *)src )
    _libc_free(v11, v5);
  return v3;
}
