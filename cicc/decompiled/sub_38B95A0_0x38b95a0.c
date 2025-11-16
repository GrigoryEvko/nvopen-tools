// Function: sub_38B95A0
// Address: 0x38b95a0
//
void __fastcall sub_38B95A0(__int64 a1, __int64 a2, int a3, __int64 a4, char a5)
{
  bool v9; // zf
  char v10; // al
  char *v11; // rsi
  size_t v12; // r15
  char *v13; // r9
  char v14; // al
  int v15; // edx
  char *v16; // rdi
  void *v17; // rdi
  size_t v18; // r15
  void *v19; // rsi
  size_t v20; // rdx
  char *v21; // rsi
  __int64 v22; // rdi
  size_t v23; // rax
  unsigned int v24; // eax
  __int64 v25; // rcx
  __int64 v26; // [rsp+0h] [rbp-150h]
  __int64 v27; // [rsp+8h] [rbp-148h]
  char *v28; // [rsp+8h] [rbp-148h]
  char *v29; // [rsp+8h] [rbp-148h]
  char *v30; // [rsp+8h] [rbp-148h]
  void *src; // [rsp+10h] [rbp-140h] BYREF
  size_t n; // [rsp+18h] [rbp-138h]
  _BYTE v33[304]; // [rsp+20h] [rbp-130h] BYREF

  v9 = *(_BYTE *)(a2 + 17) == 1;
  src = v33;
  n = 0x10000000000LL;
  if ( v9 )
  {
    v10 = *(_BYTE *)(a2 + 16);
    if ( v10 == 1 )
    {
LABEL_25:
      v14 = MEMORY[0];
      if ( MEMORY[0] == 1 )
        goto LABEL_15;
      v13 = 0;
      v12 = 0;
      goto LABEL_6;
    }
    v11 = *(char **)a2;
    switch ( v10 )
    {
      case 3:
        if ( !v11 )
          goto LABEL_25;
        v26 = a4;
        v23 = strlen(v11);
        a4 = v26;
        v12 = v23;
        v13 = v11;
        break;
      case 4:
      case 5:
        v13 = *(char **)v11;
        v12 = *((_QWORD *)v11 + 1);
        v14 = **(_BYTE **)v11;
        if ( v14 == 1 )
          goto LABEL_28;
        goto LABEL_6;
      case 6:
        v12 = *((unsigned int *)v11 + 2);
        v13 = *(char **)v11;
        break;
      default:
        goto LABEL_4;
    }
  }
  else
  {
LABEL_4:
    v27 = a4;
    sub_16E2F40(a2, (__int64)&src);
    v12 = (unsigned int)n;
    v13 = (char *)src;
    a4 = v27;
  }
  v14 = *v13;
  if ( *v13 != 1 )
  {
LABEL_6:
    v15 = *(_DWORD *)(a4 + 16);
    if ( (unsigned int)(v15 - 3) <= 1 && v14 == 63 )
      a5 = 0;
    if ( a3 == 1 )
    {
      switch ( v15 )
      {
        case 0:
          v16 = *(char **)(a1 + 24);
          break;
        case 1:
        case 3:
          v20 = 2;
          v21 = ".L";
          goto LABEL_35;
        case 2:
        case 4:
          v20 = 1;
          v21 = "L";
          goto LABEL_35;
        case 5:
          v20 = 1;
          v21 = "$";
LABEL_35:
          v22 = *(_QWORD *)(a1 + 24);
          if ( *(_QWORD *)(a1 + 16) - v22 >= v20 )
          {
            v24 = 0;
            do
            {
              v25 = v24++;
              *(_BYTE *)(v22 + v25) = v21[v25];
            }
            while ( v24 < (unsigned int)v20 );
            v16 = (char *)(v20 + *(_QWORD *)(a1 + 24));
            *(_QWORD *)(a1 + 24) = v16;
          }
          else
          {
            v30 = v13;
            sub_16E7EE0(a1, v21, v20);
            v16 = *(char **)(a1 + 24);
            v13 = v30;
          }
          break;
        default:
          ++*(_DWORD *)(v12 + 16);
          BUG();
      }
    }
    else
    {
      v16 = *(char **)(a1 + 24);
      if ( a3 == 2 && v15 == 2 )
      {
        if ( *(char **)(a1 + 16) == v16 )
        {
          v28 = v13;
          sub_16E7EE0(a1, "l", 1u);
          v16 = *(char **)(a1 + 24);
          v13 = v28;
        }
        else
        {
          *v16 = 108;
          v16 = (char *)(*(_QWORD *)(a1 + 24) + 1LL);
          *(_QWORD *)(a1 + 24) = v16;
        }
      }
    }
    if ( a5 )
    {
      if ( *(_QWORD *)(a1 + 16) <= (unsigned __int64)v16 )
      {
        v29 = v13;
        sub_16E7DE0(a1, a5);
        v16 = *(char **)(a1 + 24);
        v13 = v29;
      }
      else
      {
        *(_QWORD *)(a1 + 24) = v16 + 1;
        *v16 = a5;
        v16 = *(char **)(a1 + 24);
      }
    }
    if ( *(_QWORD *)(a1 + 16) - (_QWORD)v16 < v12 )
    {
      sub_16E7EE0(a1, v13, v12);
    }
    else if ( v12 )
    {
      memcpy(v16, v13, v12);
      *(_QWORD *)(a1 + 24) += v12;
    }
    goto LABEL_15;
  }
LABEL_28:
  if ( v12 )
  {
    v17 = *(void **)(a1 + 24);
    v18 = v12 - 1;
    v19 = v13 + 1;
    if ( v18 <= *(_QWORD *)(a1 + 16) - (_QWORD)v17 )
    {
      if ( v18 )
      {
        memcpy(v17, v19, v18);
        *(_QWORD *)(a1 + 24) += v18;
      }
    }
    else
    {
      sub_16E7EE0(a1, (char *)v19, v18);
    }
  }
LABEL_15:
  if ( src != v33 )
    _libc_free((unsigned __int64)src);
}
