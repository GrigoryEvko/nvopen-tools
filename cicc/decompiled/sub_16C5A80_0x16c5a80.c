// Function: sub_16C5A80
// Address: 0x16c5a80
//
__int64 __fastcall sub_16C5A80(__int64 a1, int *a2, int a3, int a4, char a5, unsigned int a6)
{
  unsigned int v6; // r13d
  const char *v8; // r12
  int *v9; // r15
  int v10; // eax
  __int64 v11; // rdx
  int *v12; // rcx
  __int64 v13; // r8
  unsigned int v14; // ebx
  unsigned __int64 v16[2]; // [rsp+10h] [rbp-C0h] BYREF
  _BYTE v17[176]; // [rsp+20h] [rbp-B0h] BYREF

  v6 = 0;
  if ( a4 != 1 )
  {
    v6 = 1;
    if ( a4 != 2 )
      v6 = 2 * (a4 == 3);
  }
  if ( (a5 & 2) != 0 )
  {
    v6 |= 0x440u;
  }
  else if ( a3 == 1 )
  {
    LOBYTE(v6) = v6 | 0xC0;
  }
  else if ( a3 )
  {
    if ( a3 == 3 )
      v6 |= 0x40u;
  }
  else
  {
    v6 |= 0x240u;
  }
  v16[0] = (unsigned __int64)v17;
  if ( (a5 & 8) == 0 )
    v6 |= 0x80000u;
  v16[1] = 0x8000000000LL;
  v8 = (const char *)sub_16E32E0(a1, v16);
  v9 = __errno_location();
  while ( 1 )
  {
    *v9 = 0;
    v10 = open(v8, v6, a6);
    if ( v10 != -1 )
      break;
    if ( *v9 != 4 )
    {
      *a2 = -1;
      goto LABEL_20;
    }
  }
  v12 = a2;
  *a2 = v10;
  if ( v10 >= 0 )
  {
    v14 = 0;
    sub_2241E40(v8, v6, v11, a2, v13);
    goto LABEL_16;
  }
LABEL_20:
  sub_2241E50(v8, v6, v11, v12, v13);
  v14 = *v9;
LABEL_16:
  if ( (_BYTE *)v16[0] != v17 )
    _libc_free(v16[0]);
  return v14;
}
