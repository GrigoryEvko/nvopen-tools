// Function: sub_38DD5A0
// Address: 0x38dd5a0
//
void __fastcall sub_38DD5A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  bool v3; // zf
  void (__fastcall *v4)(__int64 *, unsigned __int64, size_t); // r13
  char v5; // al
  const char *v6; // r14
  size_t v7; // rdx
  unsigned __int64 v8; // rsi
  _BYTE *v9; // [rsp+0h] [rbp-B0h] BYREF
  __int64 v10; // [rsp+8h] [rbp-A8h]
  _BYTE v11[160]; // [rsp+10h] [rbp-A0h] BYREF

  v10 = 0x8000000000LL;
  v2 = *a1;
  v3 = *(_BYTE *)(a2 + 17) == 1;
  v9 = v11;
  v4 = *(void (__fastcall **)(__int64 *, unsigned __int64, size_t))(v2 + 32);
  if ( v3 )
  {
    v5 = *(_BYTE *)(a2 + 16);
    if ( v5 == 1 )
    {
      v7 = 0;
      v8 = 0;
    }
    else
    {
      v6 = *(const char **)a2;
      switch ( v5 )
      {
        case 3:
          v7 = 0;
          if ( v6 )
            v7 = strlen(*(const char **)a2);
          v8 = (unsigned __int64)v6;
          break;
        case 4:
        case 5:
          v8 = *(_QWORD *)v6;
          v7 = *((_QWORD *)v6 + 1);
          break;
        case 6:
          v7 = *((unsigned int *)v6 + 2);
          v8 = *(_QWORD *)v6;
          break;
        default:
          goto LABEL_4;
      }
    }
  }
  else
  {
LABEL_4:
    sub_16E2F40(a2, (__int64)&v9);
    v7 = (unsigned int)v10;
    v8 = (unsigned __int64)v9;
  }
  v4(a1, v8, v7);
  if ( v9 != v11 )
    _libc_free((unsigned __int64)v9);
}
