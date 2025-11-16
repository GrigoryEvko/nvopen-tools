// Function: sub_16C2E90
// Address: 0x16c2e90
//
__int64 __fastcall sub_16C2E90(__int64 a1, __int64 a2, unsigned __int64 a3, char a4)
{
  bool v6; // zf
  char v7; // al
  const char *v8; // rcx
  size_t v9; // rax
  const char *v10; // rdx
  const char *v12; // [rsp+8h] [rbp-148h]
  const char *v13; // [rsp+10h] [rbp-140h] BYREF
  __int64 v14; // [rsp+18h] [rbp-138h]
  _BYTE v15[304]; // [rsp+20h] [rbp-130h] BYREF

  v6 = *(_BYTE *)(a2 + 17) == 1;
  v13 = v15;
  v14 = 0x10000000000LL;
  if ( v6 )
  {
    v7 = *(_BYTE *)(a2 + 16);
    if ( v7 == 1 )
    {
LABEL_6:
      sub_16C2DE0(a1, a2, a3, a4, 0);
      goto LABEL_7;
    }
    v8 = *(const char **)a2;
    switch ( v7 )
    {
      case 3:
        if ( !v8 )
          goto LABEL_6;
        v12 = *(const char **)a2;
        v9 = strlen(*(const char **)a2);
        v10 = v12;
        break;
      case 4:
      case 5:
        v10 = *(const char **)v8;
        v9 = *((_QWORD *)v8 + 1);
        break;
      case 6:
        v9 = *((unsigned int *)v8 + 2);
        v10 = *(const char **)v8;
        break;
      default:
        goto LABEL_4;
    }
  }
  else
  {
LABEL_4:
    sub_16E2F40(a2, &v13);
    v9 = (unsigned int)v14;
    v10 = v13;
  }
  if ( v9 != 1 || *v10 != 45 )
    goto LABEL_6;
  sub_16C2E10(a1);
LABEL_7:
  if ( v13 != v15 )
    _libc_free((unsigned __int64)v13);
  return a1;
}
