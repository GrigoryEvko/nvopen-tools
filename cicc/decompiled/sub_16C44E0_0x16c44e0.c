// Function: sub_16C44E0
// Address: 0x16c44e0
//
__int64 __fastcall sub_16C44E0(__int64 a1, unsigned int a2)
{
  bool v3; // zf
  char v4; // al
  const char *v5; // r13
  size_t v6; // rsi
  unsigned __int8 *v7; // rdi
  __int64 v8; // rdx
  unsigned __int8 *v10; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v11; // [rsp+8h] [rbp-A8h]
  _BYTE v12[160]; // [rsp+10h] [rbp-A0h] BYREF

  v3 = *(_BYTE *)(a1 + 17) == 1;
  v10 = v12;
  v11 = 0x8000000000LL;
  if ( v3 )
  {
    v4 = *(_BYTE *)(a1 + 16);
    if ( v4 == 1 )
    {
      v6 = 0;
      v7 = 0;
    }
    else
    {
      v5 = *(const char **)a1;
      switch ( v4 )
      {
        case 3:
          v6 = 0;
          if ( v5 )
            v6 = strlen(*(const char **)a1);
          v7 = (unsigned __int8 *)v5;
          break;
        case 4:
        case 5:
          v7 = *(unsigned __int8 **)v5;
          v6 = *((_QWORD *)v5 + 1);
          break;
        case 6:
          v6 = *((unsigned int *)v5 + 2);
          v7 = *(unsigned __int8 **)v5;
          break;
        default:
          goto LABEL_4;
      }
    }
  }
  else
  {
LABEL_4:
    sub_16E2F40(a1, &v10);
    v6 = (unsigned int)v11;
    v7 = v10;
  }
  sub_16C4220(v7, v6, a2);
  LOBYTE(a2) = v8 != 0;
  if ( v10 != v12 )
    _libc_free((unsigned __int64)v10);
  return a2;
}
