// Function: sub_23AF980
// Address: 0x23af980
//
__int64 __fastcall sub_23AF980(__int64 a1, _BYTE *a2, unsigned __int64 a3)
{
  unsigned __int64 v4; // r14
  _BYTE *v5; // rbx
  unsigned __int64 v6; // rbx
  size_t v7; // rdx
  char *v8; // rsi
  unsigned __int64 v9; // r14
  _BYTE *v10; // r15
  char *v11; // rsi
  char *v13; // [rsp+10h] [rbp-50h] BYREF
  size_t v14; // [rsp+18h] [rbp-48h]
  _BYTE v15[64]; // [rsp+20h] [rbp-40h] BYREF

  v4 = a3;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  if ( !a3 )
    goto LABEL_17;
LABEL_2:
  v5 = a2;
  while ( (*v5 & 0xFD) != 0x3C )
  {
    if ( &a2[v4] == ++v5 )
    {
      v6 = v4;
      goto LABEL_7;
    }
  }
  v6 = v5 - a2;
  if ( v6 > v4 )
    v6 = v4;
LABEL_7:
  if ( a2 )
  {
LABEL_8:
    v13 = v15;
    sub_23AE760((__int64 *)&v13, a2, (__int64)&a2[v6]);
    v7 = v14;
    v8 = v13;
    goto LABEL_9;
  }
  while ( 1 )
  {
    v13 = v15;
    v8 = v15;
    v7 = 0;
    v14 = 0;
    v15[0] = 0;
LABEL_9:
    sub_2241490((unsigned __int64 *)a1, v8, v7);
    if ( v13 != v15 )
      j_j___libc_free_0((unsigned __int64)v13);
    if ( v4 < v6 )
      return a1;
    v9 = v4 - v6;
    v10 = &a2[v6];
    if ( !v9 )
      return a1;
    v11 = "&gt;";
    if ( *v10 == 60 )
      v11 = "&lt;";
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8)) <= 3 )
      sub_4262D8((__int64)"basic_string::append");
    v4 = v9 - 1;
    a2 = v10 + 1;
    sub_2241490((unsigned __int64 *)a1, v11, 4u);
    if ( v4 )
      goto LABEL_2;
LABEL_17:
    v6 = 0;
    if ( a2 )
      goto LABEL_8;
  }
}
