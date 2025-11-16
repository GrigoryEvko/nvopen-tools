// Function: sub_16C2500
// Address: 0x16c2500
//
_QWORD *__fastcall sub_16C2500(_QWORD *a1, unsigned __int64 a2, __int64 a3)
{
  bool v4; // zf
  char v5; // al
  const char *v6; // r15
  size_t v7; // rdx
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // r12
  unsigned __int64 v10; // rdi
  __int64 v12; // rax
  size_t v13; // rdx
  _QWORD *v14; // r15
  _BYTE *v15; // rdx
  size_t n; // [rsp+0h] [rbp-150h]
  void *src; // [rsp+8h] [rbp-148h]
  _BYTE *v18; // [rsp+10h] [rbp-140h] BYREF
  __int64 v19; // [rsp+18h] [rbp-138h]
  _BYTE v20[304]; // [rsp+20h] [rbp-130h] BYREF

  v4 = *(_BYTE *)(a3 + 17) == 1;
  v18 = v20;
  v19 = 0x10000000000LL;
  if ( v4 )
  {
    v5 = *(_BYTE *)(a3 + 16);
    if ( v5 == 1 )
    {
      v9 = 32;
      v7 = 0;
      v8 = 0;
    }
    else
    {
      v6 = *(const char **)a3;
      switch ( v5 )
      {
        case 3:
          if ( v6 )
          {
            v7 = strlen(*(const char **)a3);
            v9 = (v7 + 40) & 0xFFFFFFFFFFFFFFF0LL;
          }
          else
          {
            v9 = 32;
            v7 = 0;
          }
          v8 = (unsigned __int64)v6;
          break;
        case 4:
        case 5:
          v7 = *((_QWORD *)v6 + 1);
          v8 = *(_QWORD *)v6;
          v9 = (v7 + 40) & 0xFFFFFFFFFFFFFFF0LL;
          break;
        case 6:
          v7 = *((unsigned int *)v6 + 2);
          v8 = *(_QWORD *)v6;
          v9 = (v7 + 40) & 0xFFFFFFFFFFFFFFF0LL;
          break;
        default:
          goto LABEL_4;
      }
    }
  }
  else
  {
LABEL_4:
    sub_16E2F40(a3, &v18);
    v7 = (unsigned int)v19;
    v8 = (unsigned __int64)v18;
    v9 = ((unsigned int)v19 + 40LL) & 0xFFFFFFFFFFFFFFF0LL;
  }
  v10 = v9 + a2 + 1;
  if ( a2 < v10 && (n = v7, src = (void *)v8, v12 = sub_2207800(v10, &unk_435FF63), v13 = n, (v14 = (_QWORD *)v12) != 0) )
  {
    if ( n )
    {
      memcpy((void *)(v12 + 24), src, n);
      v13 = n;
    }
    *((_BYTE *)v14 + v13 + 24) = 0;
    v15 = (char *)v14 + v9 + a2;
    *v15 = 0;
    *v14 = off_4985080;
    sub_16C2440((__int64)v14, (__int64)v14 + v9, (__int64)v15);
    *a1 = v14;
  }
  else
  {
    *a1 = 0;
  }
  if ( v18 != v20 )
    _libc_free((unsigned __int64)v18);
  return a1;
}
