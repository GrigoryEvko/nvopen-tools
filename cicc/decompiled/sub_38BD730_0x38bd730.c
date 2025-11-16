// Function: sub_38BD730
// Address: 0x38bd730
//
__int64 __fastcall sub_38BD730(__int64 a1, __int64 a2)
{
  bool v2; // zf
  char v3; // al
  const char *v4; // r13
  size_t v5; // rdx
  unsigned __int8 *v6; // rsi
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r13
  unsigned __int8 *v12; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v13; // [rsp+8h] [rbp-A8h]
  _BYTE v14[160]; // [rsp+10h] [rbp-A0h] BYREF

  v2 = *(_BYTE *)(a2 + 17) == 1;
  v12 = v14;
  v13 = 0x8000000000LL;
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
          v6 = (unsigned __int8 *)v4;
          break;
        case 4:
        case 5:
          v6 = *(unsigned __int8 **)v4;
          v5 = *((_QWORD *)v4 + 1);
          break;
        case 6:
          v5 = *((unsigned int *)v4 + 2);
          v6 = *(unsigned __int8 **)v4;
          break;
        default:
          goto LABEL_4;
      }
    }
  }
  else
  {
LABEL_4:
    sub_16E2F40(a2, (__int64)&v12);
    v5 = (unsigned int)v13;
    v6 = v12;
  }
  v7 = sub_16D1B30((__int64 *)(a1 + 568), v6, v5);
  if ( v7 == -1 || (v8 = *(_QWORD *)(a1 + 568), v9 = v8 + 8LL * v7, v9 == v8 + 8LL * *(unsigned int *)(a1 + 576)) )
    v10 = 0;
  else
    v10 = *(_QWORD *)(*(_QWORD *)v9 + 8LL);
  if ( v12 != v14 )
    _libc_free((unsigned __int64)v12);
  return v10;
}
