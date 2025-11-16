// Function: sub_16CF5C0
// Address: 0x16cf5c0
//
__int64 __fastcall sub_16CF5C0(_QWORD *a1, __int16 a2)
{
  _QWORD *v2; // rcx
  unsigned __int64 v4; // rbx
  _BYTE *v5; // rsi
  _QWORD *v6; // r13
  __int64 v7; // r14
  __int64 v8; // rsi
  __int64 v9; // rax
  _WORD *v10; // rcx
  __int64 v11; // rsi
  _WORD *v12; // rdx
  __int64 v14; // rax
  __int64 v15; // r15
  __int16 *v16; // rax
  _QWORD *v17; // [rsp+0h] [rbp-50h]
  __int16 *v18; // [rsp+8h] [rbp-48h]
  __int16 v19; // [rsp+1Eh] [rbp-32h] BYREF

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
  a1[1] = v14 | 4;
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
          sub_C8FDD0((__int64)v6, v5, v16);
          v5 = (_BYTE *)v6[1];
          v2 = v17;
          v16 = v18;
        }
        else
        {
          if ( v5 )
          {
            *(_WORD *)v5 = v4;
            v5 = (_BYTE *)v6[1];
          }
          v5 += 2;
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
  v9 = v8 >> 1;
  if ( v8 <= 0 )
    return 1;
  v10 = (_WORD *)*v6;
  do
  {
    while ( 1 )
    {
      v11 = v9 >> 1;
      v12 = (_WORD *)((char *)v10 + (v9 & 0xFFFFFFFFFFFFFFFELL));
      if ( (unsigned __int16)(a2 - v7) <= *v12 )
        break;
      v10 = v12 + 1;
      v9 = v9 - v11 - 1;
      if ( v9 <= 0 )
        return (unsigned int)(((__int64)v10 - *v6) >> 1) + 1;
    }
    v9 >>= 1;
  }
  while ( v11 > 0 );
  return (unsigned int)(((__int64)v10 - *v6) >> 1) + 1;
}
