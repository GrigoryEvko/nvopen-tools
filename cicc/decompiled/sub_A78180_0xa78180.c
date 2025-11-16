// Function: sub_A78180
// Address: 0xa78180
//
unsigned __int64 __fastcall sub_A78180(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned __int64 a3,
        _QWORD *a4,
        unsigned __int64 a5)
{
  unsigned __int64 i; // rax
  unsigned int v8; // ebx
  __int64 v9; // rdx
  unsigned __int64 *v10; // rsi
  unsigned __int64 v11; // r12
  unsigned __int64 v13; // r9
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r9
  size_t v17; // r14
  __int64 v18; // rbx
  unsigned __int64 v19; // [rsp+8h] [rbp-98h]
  unsigned __int64 v20; // [rsp+18h] [rbp-88h]
  unsigned __int64 *v21; // [rsp+20h] [rbp-80h] BYREF
  __int64 v22; // [rsp+28h] [rbp-78h]
  _QWORD v23[14]; // [rsp+30h] [rbp-70h] BYREF

  for ( i = a5; ; --i )
  {
    if ( !i )
      goto LABEL_5;
    if ( a4[i - 1] )
      break;
  }
  v8 = i + 2;
  if ( (_DWORD)i == -2 )
  {
LABEL_5:
    v8 = 2;
    if ( !a3 )
    {
      v8 = 1;
      if ( !a2 )
        return 0;
    }
    v21 = v23;
    v22 = 0x800000000LL;
LABEL_7:
    v23[0] = a2;
    v9 = (unsigned int)(v22 + 1);
    LODWORD(v22) = v22 + 1;
    if ( v8 == 1 )
      goto LABEL_8;
    v16 = v9 + 1;
    if ( v9 + 1 <= (unsigned __int64)HIDWORD(v22) )
    {
LABEL_28:
      v21[v9] = a3;
      v9 = (unsigned int)(v22 + 1);
      LODWORD(v22) = v22 + 1;
      if ( v8 != 2 )
        goto LABEL_18;
LABEL_8:
      v10 = v21;
      goto LABEL_9;
    }
LABEL_30:
    v20 = a5;
    sub_C8D5F0(&v21, v23, v16, 8);
    v9 = (unsigned int)v22;
    a5 = v20;
    goto LABEL_28;
  }
  v21 = v23;
  v22 = 0x800000000LL;
  if ( v8 <= 8 )
    goto LABEL_7;
  v19 = a5;
  sub_C8D5F0(&v21, v23, v8, 8);
  v13 = a2;
  v14 = (unsigned int)v22 + 1LL;
  a5 = v19;
  if ( v14 > HIDWORD(v22) )
  {
    sub_C8D5F0(&v21, v23, v14, 8);
    v13 = a2;
    a5 = v19;
  }
  v21[(unsigned int)v22] = v13;
  v15 = (unsigned int)(v22 + 1);
  v16 = v15 + 1;
  LODWORD(v22) = v22 + 1;
  if ( HIDWORD(v22) < (unsigned __int64)(v15 + 1) )
    goto LABEL_30;
  v21[v15] = a3;
  v9 = (unsigned int)(v22 + 1);
  LODWORD(v22) = v22 + 1;
LABEL_18:
  if ( v8 - 2 <= a5 )
    a5 = v8 - 2;
  v17 = 8 * a5;
  v18 = (__int64)(8 * a5) >> 3;
  if ( v9 + v18 > (unsigned __int64)HIDWORD(v22) )
  {
    sub_C8D5F0(&v21, v23, v9 + v18, 8);
    v9 = (unsigned int)v22;
  }
  v10 = v21;
  if ( v17 )
  {
    memcpy(&v21[v9], a4, v17);
    LODWORD(v9) = v22;
    v10 = v21;
  }
  LODWORD(v22) = v9 + v18;
  v9 = (unsigned int)(v9 + v18);
LABEL_9:
  v11 = sub_A77EC0(a1, v10, v9);
  if ( v21 != v23 )
    _libc_free(v21, v10);
  return v11;
}
