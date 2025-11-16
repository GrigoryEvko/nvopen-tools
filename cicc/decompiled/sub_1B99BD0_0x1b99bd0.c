// Function: sub_1B99BD0
// Address: 0x1b99bd0
//
unsigned __int64 __fastcall sub_1B99BD0(
        unsigned int *a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned int *a5,
        int a6)
{
  __int64 v8; // rbx
  unsigned int *v9; // rax
  unsigned int *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  _BYTE *v14; // rdi
  int v15; // eax
  size_t v16; // rdx
  _QWORD *v17; // r13
  unsigned __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // r8d
  int v22; // r9d
  unsigned __int64 result; // rax
  int v24; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v25; // [rsp+10h] [rbp-60h]
  unsigned __int64 v26; // [rsp+18h] [rbp-58h] BYREF
  _BYTE *v27; // [rsp+20h] [rbp-50h] BYREF
  __int64 v28; // [rsp+28h] [rbp-48h]
  _BYTE s[64]; // [rsp+30h] [rbp-40h] BYREF

  v8 = a3;
  v9 = (unsigned int *)*((_QWORD *)a1 + 3);
  v26 = a2;
  if ( !v9 )
    goto LABEL_8;
  a5 = a1 + 4;
  v10 = a1 + 4;
  do
  {
    while ( 1 )
    {
      v11 = *((_QWORD *)v9 + 2);
      v12 = *((_QWORD *)v9 + 3);
      if ( *((_QWORD *)v9 + 4) >= a2 )
        break;
      v9 = (unsigned int *)*((_QWORD *)v9 + 3);
      if ( !v12 )
        goto LABEL_6;
    }
    v10 = v9;
    v9 = (unsigned int *)*((_QWORD *)v9 + 2);
  }
  while ( v11 );
LABEL_6:
  if ( a5 != v10 && *((_QWORD *)v10 + 4) <= a2 )
  {
    v17 = a1 + 2;
  }
  else
  {
LABEL_8:
    v13 = *a1;
    v27 = s;
    v14 = s;
    v28 = 0x200000000LL;
    v15 = v13;
    if ( (unsigned int)v13 > 2 )
    {
      v24 = v13;
      v25 = v13;
      sub_16CD150((__int64)&v27, s, v13, 8, (int)a5, a6);
      v14 = v27;
      v15 = v24;
      v13 = v25;
    }
    v16 = 8 * v13;
    LODWORD(v28) = v15;
    if ( v16 )
      memset(v14, 0, v16);
    v17 = a1 + 2;
    v18 = sub_1B99AC0(v17, &v26);
    sub_1B8E680((__int64)v18, (__int64)&v27, v19, v20, v21, v22);
    if ( v27 != s )
      _libc_free((unsigned __int64)v27);
  }
  result = *sub_1B99AC0(v17, &v26);
  *(_QWORD *)(result + 8 * v8) = a4;
  return result;
}
