// Function: sub_16CF740
// Address: 0x16cf740
//
__int64 __fastcall sub_16CF740(_QWORD *a1, int a2)
{
  _QWORD *v2; // rcx
  unsigned __int64 v4; // rbx
  _BYTE *v5; // rsi
  _QWORD *v6; // r13
  __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rax
  _DWORD *v10; // rsi
  __int64 v11; // rdx
  _DWORD *v12; // rcx
  __int64 v14; // rax
  __int64 v15; // r15
  int *v16; // rax
  _QWORD *v17; // [rsp+0h] [rbp-50h]
  int *v18; // [rsp+8h] [rbp-48h]
  int v19; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = a1;
  v4 = a1[1] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 )
  {
    v5 = *(_BYTE **)(v4 + 8);
    v6 = (_QWORD *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
LABEL_3:
    v7 = *(_QWORD *)(*v2 + 8LL);
    goto LABEL_4;
  }
  v14 = sub_22077B0(24);
  v2 = a1;
  v6 = (_QWORD *)v14;
  if ( v14 )
  {
    *(_QWORD *)v14 = 0;
    v5 = 0;
    *(_QWORD *)(v14 + 8) = 0;
    *(_QWORD *)(v14 + 16) = 0;
  }
  else
  {
    v5 = (_BYTE *)MEMORY[8];
  }
  a1[1] = v14 | 2;
  v7 = *(_QWORD *)(*a1 + 8LL);
  v15 = *(_QWORD *)(*a1 + 16LL) - v7;
  if ( v15 )
  {
    v16 = &v19;
    do
    {
      if ( *(_BYTE *)(v7 + v4) == 10 )
      {
        v19 = v4;
        if ( (_BYTE *)v6[2] == v5 )
        {
          v17 = v2;
          v18 = v16;
          sub_C88AB0((__int64)v6, v5, v16);
          v5 = (_BYTE *)v6[1];
          v2 = v17;
          v16 = v18;
        }
        else
        {
          if ( v5 )
          {
            *(_DWORD *)v5 = v4;
            v5 = (_BYTE *)v6[1];
          }
          v5 += 4;
          v6[1] = v5;
        }
      }
      ++v4;
    }
    while ( v4 != v15 );
    goto LABEL_3;
  }
LABEL_4:
  v8 = (__int64)&v5[-*v6];
  v9 = v8 >> 2;
  if ( v8 <= 0 )
    return 1;
  v10 = (_DWORD *)*v6;
  do
  {
    while ( 1 )
    {
      v11 = v9 >> 1;
      v12 = &v10[v9 >> 1];
      if ( (unsigned int)(a2 - v7) <= *v12 )
        break;
      v10 = v12 + 1;
      v9 = v9 - v11 - 1;
      if ( v9 <= 0 )
        return (unsigned int)(((__int64)v10 - *v6) >> 2) + 1;
    }
    v9 >>= 1;
  }
  while ( v11 > 0 );
  return (unsigned int)(((__int64)v10 - *v6) >> 2) + 1;
}
