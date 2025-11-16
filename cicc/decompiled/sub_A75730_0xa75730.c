// Function: sub_A75730
// Address: 0xa75730
//
unsigned __int64 __fastcall sub_A75730(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  bool v7; // zf
  char *v8; // r13
  size_t v9; // r15
  _QWORD *v10; // rax
  unsigned __int64 v11; // rax
  _BYTE *v12; // rdi
  _QWORD *v13; // rdi
  unsigned __int64 v14; // [rsp+0h] [rbp-80h] BYREF
  _BYTE *v15; // [rsp+8h] [rbp-78h] BYREF
  char v16; // [rsp+24h] [rbp-5Ch] BYREF
  _BYTE v17[11]; // [rsp+25h] [rbp-5Bh] BYREF
  _QWORD *v18; // [rsp+30h] [rbp-50h] BYREF
  size_t v19; // [rsp+38h] [rbp-48h]
  _QWORD v20[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = a2;
  result = sub_B2D7E0(a1, "min-legal-vector-width", 22);
  v14 = result;
  if ( result )
  {
    v5 = sub_A72240((__int64 *)&v14);
    v7 = (unsigned __int8)sub_C93C90(v5, v6, 0, &v18) == 0;
    result = 0;
    if ( v7 )
      result = (unsigned __int64)v18;
    if ( a2 > result )
    {
      if ( !a2 )
      {
        v16 = 48;
        v8 = &v16;
        v18 = v20;
LABEL_8:
        v9 = 1;
        LOBYTE(v20[0]) = *v8;
        v10 = v20;
        goto LABEL_9;
      }
      v8 = v17;
      do
      {
        *--v8 = v3 % 0xA + 48;
        v11 = v3;
        v3 /= 0xAu;
      }
      while ( v11 > 9 );
      v12 = (_BYTE *)(v17 - v8);
      v18 = v20;
      v9 = v17 - v8;
      v15 = (_BYTE *)(v17 - v8);
      if ( (unsigned __int64)(v17 - v8) <= 0xF )
      {
        if ( v12 == (_BYTE *)1 )
          goto LABEL_8;
        if ( !v12 )
        {
          v10 = v20;
LABEL_9:
          v19 = v9;
          *((_BYTE *)v10 + v9) = 0;
          result = sub_B2CD60(a1, "min-legal-vector-width", 22, v18, v19);
          if ( v18 != v20 )
            return j_j___libc_free_0(v18, v20[0] + 1LL);
          return result;
        }
        v13 = v20;
      }
      else
      {
        v18 = (_QWORD *)sub_22409D0(&v18, &v15, 0);
        v13 = v18;
        v20[0] = v15;
      }
      memcpy(v13, v8, v9);
      v9 = (size_t)v15;
      v10 = v18;
      goto LABEL_9;
    }
  }
  return result;
}
