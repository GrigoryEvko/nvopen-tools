// Function: sub_ED1AF0
// Address: 0xed1af0
//
_QWORD *__fastcall sub_ED1AF0(_QWORD *a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v5; // rdx
  __int64 v7; // rbx
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rcx
  unsigned __int64 v15; // r12
  _QWORD *v16; // r14
  _BYTE *v17; // rcx
  unsigned __int64 v18; // rdx
  char i; // al
  __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  __int64 v22; // rcx
  _BYTE v25[32]; // [rsp+20h] [rbp-110h] BYREF
  _QWORD *v26; // [rsp+40h] [rbp-F0h] BYREF
  unsigned __int64 v27; // [rsp+48h] [rbp-E8h]
  _QWORD v28[2]; // [rsp+50h] [rbp-E0h] BYREF
  _QWORD v29[3]; // [rsp+60h] [rbp-D0h] BYREF
  char v30; // [rsp+78h] [rbp-B8h] BYREF

  v5 = 32 * a3;
  v7 = a2 + v5;
  v26 = v28;
  v27 = 0;
  LOBYTE(v28[0]) = 0;
  if ( a2 + v5 == a2 )
  {
    v16 = v28;
    v15 = 0;
    i = 0;
    v17 = v25;
  }
  else
  {
    v9 = (v5 >> 5) - 1;
    v10 = a2;
    do
    {
      v9 += *(_QWORD *)(v10 + 8);
      v10 += 32;
    }
    while ( v7 != v10 );
    v11 = a2 + 32;
    sub_2240E30(&v26, v9);
    sub_2241490(&v26, *(_QWORD *)(v11 - 32), *(_QWORD *)(v11 - 24), v12);
    if ( v7 != v11 )
    {
      while ( v27 != 0x3FFFFFFFFFFFFFFFLL )
      {
        v11 += 32;
        sub_2241490(&v26, &unk_3F871B2, 1, v13);
        sub_2241490(&v26, *(_QWORD *)(v11 - 32), *(_QWORD *)(v11 - 24), v14);
        if ( v7 == v11 )
          goto LABEL_7;
      }
LABEL_17:
      sub_4262D8((__int64)"basic_string::append");
    }
LABEL_7:
    v15 = v27;
    v16 = v26;
    v17 = v25;
    v18 = v27 >> 7;
    for ( i = v27 & 0x7F; v18; v18 >>= 7 )
    {
      *v17++ = i | 0x80;
      i = v18 & 0x7F;
    }
  }
  *v17 = i;
  v20 = (_DWORD)v17 + 1 - (unsigned int)v25;
  if ( a4 )
  {
    v29[0] = &v30;
    v29[1] = 0;
    v29[2] = 128;
    sub_409306(v16, v15, v29, 9);
  }
  v25[v20] = 0;
  v21 = (unsigned int)&v25[v20] + 1 - (unsigned int)v25;
  if ( v21 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a5 + 8) )
    goto LABEL_17;
  sub_2241490(a5, v25, v21, v20);
  if ( 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a5 + 8) < v15 )
    goto LABEL_17;
  sub_2241490(a5, v16, v15, v22);
  *a1 = 1;
  if ( v26 != v28 )
    j_j___libc_free_0(v26, v28[0] + 1LL);
  return a1;
}
