// Function: sub_C8D970
// Address: 0xc8d970
//
__int64 __fastcall sub_C8D970(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  void *v4; // rcx
  int v6; // r15d
  unsigned int v7; // r13d
  unsigned __int64 v8; // r9
  char *v9; // rax
  unsigned __int64 v10; // r11
  char *v11; // rbx
  _BYTE *v12; // rdi
  char *v13; // rdx
  const void *v14; // rsi
  size_t v15; // rdx
  size_t v16; // r12
  const void *v17; // rsi
  _BYTE *v19; // rdx
  int v20; // [rsp+4h] [rbp-4Ch]
  void *dest; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22; // [rsp+10h] [rbp-40h]
  size_t v23; // [rsp+10h] [rbp-40h]

  v20 = a3;
  if ( (_DWORD)a3 )
  {
    v4 = *(void **)(a1 + 32);
    v6 = 0;
    v7 = 0;
    v8 = 0;
    while ( 1 )
    {
      dest = v4;
      v22 = v8;
      v9 = (char *)memchr((const void *)(a2 + v8), 9, a3 - v8);
      v10 = *(_QWORD *)(a1 + 24);
      v8 = v22;
      v4 = dest;
      if ( !v9 )
        goto LABEL_20;
      v11 = &v9[-a2];
      v12 = dest;
      if ( &v9[-a2] == (char *)-1LL )
        goto LABEL_20;
      if ( a3 <= v22 )
        v8 = a3;
      if ( (unsigned __int64)v11 >= v8 )
      {
        v13 = &v9[-a2];
        if ( a3 <= (unsigned __int64)v11 )
          v13 = (char *)a3;
        v14 = (const void *)(a2 + v8);
        v15 = (size_t)&v13[-v8];
        if ( v10 - (unsigned __int64)dest >= v15 )
        {
          if ( v15 )
          {
            v23 = v15;
            memcpy(dest, v14, v15);
            v19 = (_BYTE *)(*(_QWORD *)(a1 + 32) + v23);
            *(_QWORD *)(a1 + 32) = v19;
            v12 = v19;
          }
        }
        else
        {
          sub_CB6200(a1, v14, v15);
          v12 = *(_BYTE **)(a1 + 32);
        }
      }
      v6 = (_DWORD)v11 + v6 - v7;
      do
      {
        if ( *(_QWORD *)(a1 + 24) > (unsigned __int64)v12 )
        {
          *(_QWORD *)(a1 + 32) = v12 + 1;
          *v12 = 32;
        }
        else
        {
          sub_CB5D20(a1, 32);
        }
        v12 = *(_BYTE **)(a1 + 32);
        ++v6;
        v4 = v12;
      }
      while ( (v6 & 7) != 0 );
      v7 = (_DWORD)v11 + 1;
      if ( v20 == (_DWORD)v11 + 1 )
        break;
      v8 = v7;
      if ( v7 >= a3 )
      {
        v10 = *(_QWORD *)(a1 + 24);
LABEL_20:
        v12 = v4;
        if ( a3 >= v8 )
        {
          v16 = a3 - v8;
          v17 = (const void *)(v8 + a2);
          if ( v10 - (unsigned __int64)v4 >= v16 )
          {
            if ( v16 )
            {
              memcpy(v4, v17, v16);
              v10 = *(_QWORD *)(a1 + 24);
              v12 = (_BYTE *)(v16 + *(_QWORD *)(a1 + 32));
              *(_QWORD *)(a1 + 32) = v12;
            }
          }
          else
          {
            sub_CB6200(a1, v17, v16);
            v12 = *(_BYTE **)(a1 + 32);
            v10 = *(_QWORD *)(a1 + 24);
          }
        }
        goto LABEL_23;
      }
    }
    if ( *(_QWORD *)(a1 + 24) > (unsigned __int64)v12 )
      goto LABEL_24;
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 24);
    v12 = *(_BYTE **)(a1 + 32);
LABEL_23:
    if ( v10 > (unsigned __int64)v12 )
    {
LABEL_24:
      *(_QWORD *)(a1 + 32) = v12 + 1;
      *v12 = 10;
      return (__int64)(v12 + 1);
    }
  }
  return sub_CB5D20(a1, 10);
}
