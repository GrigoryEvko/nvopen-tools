// Function: sub_C83360
// Address: 0xc83360
//
__int64 __fastcall sub_C83360(__int64 a1, int *a2, int a3, int a4, char a5, unsigned int a6)
{
  unsigned int v6; // r13d
  const char *v8; // r12
  int *v9; // r15
  int v10; // eax
  __int64 v11; // rdx
  int *v12; // rcx
  __int64 v13; // r8
  unsigned int v14; // ebx
  _QWORD v16[3]; // [rsp+10h] [rbp-D0h] BYREF
  _BYTE v17[184]; // [rsp+28h] [rbp-B8h] BYREF

  v6 = 0;
  if ( a4 != 1 )
  {
    v6 = 1;
    if ( a4 != 2 )
      v6 = 2 * (a4 == 3);
  }
  if ( (a5 & 4) != 0 )
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
  v16[1] = 0;
  v16[0] = v17;
  v16[2] = 128;
  if ( (a5 & 0x10) == 0 )
    v6 |= 0x80000u;
  v8 = (const char *)sub_CA12A0(a1, v16);
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
    _libc_free(v16[0], v6);
  return v14;
}
