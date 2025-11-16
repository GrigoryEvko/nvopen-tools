// Function: sub_16C2200
// Address: 0x16c2200
//
__int64 __fastcall sub_16C2200(__int64 a1, __int64 a2)
{
  bool v2; // zf
  char v3; // al
  const char *v4; // r12
  size_t v5; // r15
  void *v6; // r14
  __int64 v7; // r12
  void *v8; // rbx
  _BYTE *v9; // rdi
  void *src; // [rsp+0h] [rbp-140h] BYREF
  size_t n; // [rsp+8h] [rbp-138h]
  _BYTE v13[304]; // [rsp+10h] [rbp-130h] BYREF

  v2 = *(_BYTE *)(a2 + 17) == 1;
  src = v13;
  n = 0x10000000000LL;
  if ( v2 )
  {
    v3 = *(_BYTE *)(a2 + 16);
    if ( v3 == 1 )
    {
      v5 = 0;
      v6 = 0;
    }
    else
    {
      v4 = *(const char **)a2;
      switch ( v3 )
      {
        case 3:
          v5 = 0;
          if ( v4 )
            v5 = strlen(*(const char **)a2);
          v6 = (void *)v4;
          break;
        case 4:
        case 5:
          v6 = *(void **)v4;
          v5 = *((_QWORD *)v4 + 1);
          break;
        case 6:
          v5 = *((unsigned int *)v4 + 2);
          v6 = *(void **)v4;
          break;
        default:
          goto LABEL_4;
      }
    }
  }
  else
  {
LABEL_4:
    sub_16E2F40(a2, &src);
    v5 = (unsigned int)n;
    v6 = src;
  }
  v7 = sub_22077B0(v5 + a1 + 1);
  v8 = (void *)(v7 + a1);
  if ( v5 )
    memcpy(v8, v6, v5);
  v9 = src;
  *((_BYTE *)v8 + v5) = 0;
  if ( v9 != v13 )
    _libc_free((unsigned __int64)v9);
  return v7;
}
