// Function: sub_1632310
// Address: 0x1632310
//
__int64 __fastcall sub_1632310(__int64 a1, __int64 a2)
{
  bool v2; // zf
  char v3; // al
  const char *v4; // r13
  size_t v5; // rdx
  unsigned __int64 v6; // rsi
  __int64 v7; // r12
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // r12
  _BYTE *v12; // [rsp+0h] [rbp-130h] BYREF
  __int64 v13; // [rsp+8h] [rbp-128h]
  _BYTE v14[288]; // [rsp+10h] [rbp-120h] BYREF

  v2 = *(_BYTE *)(a2 + 17) == 1;
  v12 = v14;
  v13 = 0x10000000000LL;
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
          v6 = (unsigned __int64)v4;
          break;
        case 4:
        case 5:
          v6 = *(_QWORD *)v4;
          v5 = *((_QWORD *)v4 + 1);
          break;
        case 6:
          v5 = *((unsigned int *)v4 + 2);
          v6 = *(_QWORD *)v4;
          break;
        default:
          goto LABEL_4;
      }
    }
  }
  else
  {
LABEL_4:
    sub_16E2F40(a2, &v12);
    v5 = (unsigned int)v13;
    v6 = (unsigned __int64)v12;
  }
  v7 = *(_QWORD *)(a1 + 272);
  v8 = sub_16D1B30(v7, v6, v5);
  if ( v8 == -1 || (v9 = *(_QWORD *)v7 + 8LL * v8, v9 == *(_QWORD *)v7 + 8LL * *(unsigned int *)(v7 + 8)) )
    v10 = 0;
  else
    v10 = *(_QWORD *)(*(_QWORD *)v9 + 8LL);
  if ( v12 != v14 )
    _libc_free((unsigned __int64)v12);
  return v10;
}
