// Function: sub_16C4E60
// Address: 0x16c4e60
//
__int64 __fastcall sub_16C4E60(__int64 a1, unsigned int a2)
{
  bool v2; // zf
  char v3; // al
  const char *v4; // r12
  size_t v5; // rax
  unsigned __int64 v6; // rdx
  unsigned int v7; // r12d
  _QWORD v9[2]; // [rsp+0h] [rbp-F0h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-E0h] BYREF
  __int16 v11; // [rsp+20h] [rbp-D0h]
  _BYTE *v12; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v13; // [rsp+38h] [rbp-B8h]
  _BYTE v14[176]; // [rsp+40h] [rbp-B0h] BYREF

  v2 = *(_BYTE *)(a1 + 17) == 1;
  v12 = v14;
  v13 = 0x8000000000LL;
  if ( v2 )
  {
    v3 = *(_BYTE *)(a1 + 16);
    if ( v3 == 1 )
    {
      v5 = 0;
      v6 = 0;
    }
    else
    {
      v4 = *(const char **)a1;
      switch ( v3 )
      {
        case 3:
          v5 = 0;
          if ( v4 )
            v5 = strlen(*(const char **)a1);
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
    sub_16E2F40(a1, &v12);
    v5 = (unsigned int)v13;
    v6 = (unsigned __int64)v12;
  }
  v9[0] = v6;
  v9[1] = v5;
  v11 = 261;
  v10[0] = v9;
  v7 = sub_16C4D60((__int64)v10, a2);
  if ( !a2 )
  {
    v10[0] = v9;
    v11 = 261;
    v7 &= sub_16C44E0((__int64)v10, 0);
  }
  if ( v12 != v14 )
    _libc_free((unsigned __int64)v12);
  return v7;
}
