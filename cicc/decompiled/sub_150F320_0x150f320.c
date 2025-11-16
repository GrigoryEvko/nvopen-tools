// Function: sub_150F320
// Address: 0x150f320
//
__int64 __fastcall sub_150F320(__int64 a1, unsigned int a2)
{
  unsigned int v3; // r8d
  __int64 v4; // r9
  unsigned __int64 v5; // r14
  unsigned __int64 v6; // r11
  unsigned int v7; // r13d
  unsigned __int64 v8; // rbx
  _QWORD *v9; // r12
  unsigned int v10; // r11d
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // r14
  char v15; // r12
  unsigned __int64 v16; // r8
  unsigned __int64 v17; // rbx
  unsigned int v18; // r15d
  unsigned __int64 v19; // rax
  _QWORD *v20; // r11
  unsigned int v21; // r8d
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rsi
  unsigned int v24; // r9d
  unsigned int v25; // r11d
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdx
  char v30; // cl
  int v31; // r8d
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 v34; // rdx
  char v35; // cl
  unsigned __int64 v37; // rax
  int v38; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v39; // [rsp+10h] [rbp-40h]
  char v40; // [rsp+1Ch] [rbp-34h]

  v3 = *(_DWORD *)(a1 + 32);
  if ( a2 > v3 )
  {
    v4 = 0;
    if ( v3 )
      v4 = *(_QWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 16);
    v6 = *(_QWORD *)(a1 + 8);
    v7 = a2 - v3;
    if ( v5 < v6 )
    {
      v8 = v5 + 8;
      v9 = (_QWORD *)(v5 + *(_QWORD *)a1);
      if ( v6 < v5 + 8 )
      {
        *(_QWORD *)(a1 + 24) = 0;
        v25 = v6 - v5;
        if ( !v25 )
        {
LABEL_31:
          *(_DWORD *)(a1 + 32) = 0;
          goto LABEL_32;
        }
        v26 = v25;
        v27 = 0;
        v28 = 0;
        do
        {
          v29 = *((unsigned __int8 *)v9 + v27);
          v30 = 8 * v27++;
          v28 |= v29 << v30;
          *(_QWORD *)(a1 + 24) = v28;
        }
        while ( v25 != v27 );
        v10 = 8 * v25;
        v8 = v5 + v26;
      }
      else
      {
        v10 = 64;
        *(_QWORD *)(a1 + 24) = *v9;
      }
      *(_QWORD *)(a1 + 16) = v8;
      *(_DWORD *)(a1 + 32) = v10;
      if ( v7 <= v10 )
      {
        v11 = *(_QWORD *)(a1 + 24);
        v12 = v11 >> v7;
        *(_QWORD *)(a1 + 24) = v11 >> v7;
        *(_DWORD *)(a1 + 32) = v3 - a2 + v10;
        v13 = v4 | ((v11 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v3 - (unsigned __int8)a2 + 64))) << v3);
        goto LABEL_9;
      }
    }
LABEL_32:
    sub_16BD130("Unexpected end of file", 1);
  }
  v37 = *(_QWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 32) = v3 - a2;
  v13 = v37 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a2));
  v12 = v37 >> a2;
  *(_QWORD *)(a1 + 24) = v12;
LABEL_9:
  v14 = (unsigned int)v13;
  v40 = a2 - 1;
  if ( _bittest((const int *)&v13, a2 - 1) )
  {
    v15 = 0;
    v38 = ~(-1 << v40);
    v14 = v38 & (unsigned int)v13;
    do
    {
      v24 = *(_DWORD *)(a1 + 32);
      v15 += v40;
      if ( a2 > v24 )
      {
        v16 = *(_QWORD *)(a1 + 8);
        if ( !v24 )
          v12 = 0;
        v17 = *(_QWORD *)(a1 + 16);
        v18 = a2 - v24;
        v39 = v12;
        if ( v17 >= v16 )
          goto LABEL_32;
        v19 = v17 + 8;
        v20 = (_QWORD *)(v17 + *(_QWORD *)a1);
        if ( v16 < v17 + 8 )
        {
          *(_QWORD *)(a1 + 24) = 0;
          v31 = v16 - v17;
          if ( !v31 )
            goto LABEL_31;
          v32 = 0;
          v33 = 0;
          do
          {
            v34 = *((unsigned __int8 *)v20 + v32);
            v35 = 8 * v32++;
            v33 |= v34 << v35;
            *(_QWORD *)(a1 + 24) = v33;
          }
          while ( v32 != v31 );
          v21 = 8 * v31;
          v19 = v17 + v32;
        }
        else
        {
          v21 = 64;
          *(_QWORD *)(a1 + 24) = *v20;
        }
        *(_QWORD *)(a1 + 16) = v19;
        *(_DWORD *)(a1 + 32) = v21;
        if ( v18 > v21 )
          goto LABEL_32;
        v22 = *(_QWORD *)(a1 + 24);
        v12 = v22 >> v18;
        *(_QWORD *)(a1 + 24) = v22 >> v18;
        *(_DWORD *)(a1 + 32) = v24 - a2 + v21;
        v23 = v39 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v24 - (unsigned __int8)a2 + 64)) & v22) << v24);
      }
      else
      {
        *(_DWORD *)(a1 + 32) = v24 - a2;
        LODWORD(v23) = v12 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)a2));
        v12 >>= a2 & 0x3F;
        *(_QWORD *)(a1 + 24) = v12;
      }
      v14 |= (unsigned __int64)((unsigned int)v23 & v38) << v15;
    }
    while ( ((unsigned int)v23 & (1 << v40)) != 0 );
  }
  return v14;
}
