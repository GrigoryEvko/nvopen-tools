// Function: sub_16CF450
// Address: 0x16cf450
//
__int64 __fastcall sub_16CF450(_QWORD *a1, char a2)
{
  _QWORD *v2; // rcx
  unsigned __int64 v3; // rbx
  _BYTE *v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // r14
  unsigned __int8 v7; // di
  __int64 v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // rdx
  unsigned __int8 *v11; // rcx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r15
  char *v16; // rdx
  _QWORD *v17; // [rsp+0h] [rbp-50h]
  char *v18; // [rsp+8h] [rbp-48h]
  char v19; // [rsp+1Fh] [rbp-31h] BYREF

  v2 = a1;
  v3 = a1[1] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 )
  {
    v4 = *(_BYTE **)(v3 + 8);
    v5 = (_QWORD *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
LABEL_3:
    v6 = *(_QWORD *)(*v2 + 8LL);
    goto LABEL_4;
  }
  v13 = (_QWORD *)sub_22077B0(24);
  v2 = a1;
  v5 = v13;
  if ( v13 )
  {
    *v13 = 0;
    v13[1] = 0;
    v13[2] = 0;
    v4 = 0;
  }
  else
  {
    v4 = (_BYTE *)MEMORY[8];
  }
  v14 = *a1;
  a1[1] = v5;
  v6 = *(_QWORD *)(v14 + 8);
  v15 = *(_QWORD *)(v14 + 16) - v6;
  if ( v15 )
  {
    v16 = &v19;
    do
    {
      if ( *(_BYTE *)(v6 + v3) == 10 )
      {
        v19 = v3;
        if ( (_BYTE *)v5[2] == v4 )
        {
          v17 = v2;
          v18 = v16;
          sub_C8FB10((__int64)v5, v4, v16);
          v4 = (_BYTE *)v5[1];
          v2 = v17;
          v16 = v18;
        }
        else
        {
          if ( v4 )
          {
            *v4 = v3;
            v4 = (_BYTE *)v5[1];
          }
          v5[1] = ++v4;
        }
      }
      ++v3;
    }
    while ( v3 != v15 );
    goto LABEL_3;
  }
LABEL_4:
  v7 = a2 - v6;
  v8 = (__int64)&v4[-*v5];
  if ( v8 <= 0 )
    return 1;
  v9 = (_BYTE *)*v5;
  do
  {
    while ( 1 )
    {
      v10 = v8 >> 1;
      v11 = &v9[v8 >> 1];
      if ( v7 <= *v11 )
        break;
      v9 = v11 + 1;
      v8 = v8 - v10 - 1;
      if ( v8 <= 0 )
        return (unsigned int)v9 - (unsigned int)*v5 + 1;
    }
    v8 >>= 1;
  }
  while ( v10 > 0 );
  return (unsigned int)v9 - (unsigned int)*v5 + 1;
}
