// Function: sub_1686CE0
// Address: 0x1686ce0
//
void __fastcall sub_1686CE0(unsigned __int64 **a1, unsigned __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 *v3; // r13
  unsigned __int64 **v4; // r15
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // r8
  __int64 v7; // rbx
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // r10
  unsigned __int64 v10; // r11
  __int64 v11; // rdx
  __int64 v12; // r9
  unsigned __int64 v13; // r15
  _QWORD *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // r9
  unsigned __int64 *v17; // rdi
  __int64 v18; // rax
  unsigned __int64 *v19; // rsi
  unsigned int v20; // ecx
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h]
  __int64 v25; // [rsp+30h] [rbp-50h]
  unsigned __int64 v26; // [rsp+38h] [rbp-48h]
  unsigned __int64 v27; // [rsp+38h] [rbp-48h]
  unsigned __int64 v28; // [rsp+40h] [rbp-40h]
  unsigned int v30; // [rsp+4Ch] [rbp-34h]

  v3 = *a1;
  if ( !a3 || !v3 )
    return;
  v4 = a1;
  v5 = a3;
  v30 = *((_DWORD *)v3 + 2);
  if ( v30 <= 0x3B )
  {
    v21 = a2 + a3 - 1;
    if ( a2 < *v3 )
      a2 = *v3;
    v22 = *v3 + (16LL << v30) - 1;
    if ( v22 > v21 )
      v22 = v21;
    if ( a2 > v22 )
      return;
    v5 = v22 - a2 + 1;
    if ( !v5 )
      return;
    goto LABEL_5;
  }
  if ( v30 <= 0x3F )
  {
LABEL_5:
    v6 = a2;
    v7 = (a2 >> v30) & 0xF;
    a2 &= ~(-1LL << v30);
    goto LABEL_6;
  }
  v6 = a2;
  v7 = 0;
LABEL_6:
  v8 = 1LL << v30;
  if ( a2 + v5 - 1 >= (1LL << v30) - 1 )
  {
    v10 = 0;
    v9 = v8 - a2;
  }
  else
  {
    v9 = v5;
    v10 = v8 - (a2 + v5);
  }
  v26 = v10;
  if ( a2 )
  {
    v28 = v9 + v6;
    if ( *((_BYTE *)v3 + v7 + 12) )
    {
      v11 = v7 + 4;
      v12 = v3[v7 + 4];
      if ( v12 )
      {
        v3[v11] = 0;
        *((_BYTE *)v3 + v7 + 12) = 0;
        v23 = v9;
        v24 = v12;
        v25 = (__int64)&v3[v11];
        sub_1686610(v25, (_BYTE *)v3 + v7 + 12, 0, v6 - a2, a2, v12, v30 - 4);
        sub_1686610(v25, (_BYTE *)v3 + v7 + 12, 0, v28, v26, v24, v30 - 4);
        v9 = v23;
      }
    }
    else
    {
      v27 = v9;
      sub_1686CE0(&v3[v7 + 4], v6, v9);
      v9 = v27;
    }
    v6 = v28;
    ++v7;
    v5 -= v9;
  }
  if ( v8 <= v5 )
  {
    v13 = v6;
    do
    {
      if ( !*((_BYTE *)v3 + v7 + 12) )
      {
        v14 = (_QWORD *)v3[v7 + 4];
        if ( v14 )
          sub_1686480(v14);
      }
      v3[v7 + 4] = 0;
      v5 -= v8;
      v13 += v8;
      *((_BYTE *)v3 + v7++ + 12) = 0;
    }
    while ( v8 <= v5 );
    v6 = v13;
    v4 = a1;
  }
  if ( v5 )
  {
    if ( *((_BYTE *)v3 + v7 + 12) )
    {
      v15 = v7 + 4;
      v16 = v3[v7 + 4];
      if ( v16 )
      {
        v3[v15] = 0;
        *((_BYTE *)v3 + v7 + 12) = 0;
        sub_1686610((__int64)&v3[v15], (_BYTE *)v3 + v7 + 12, 0, v6 + v5, v8 - v5, v16, v30 - 4);
      }
    }
    else
    {
      sub_1686CE0(&v3[v7 + 4], v6, v5);
    }
  }
  v17 = *v4;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  while ( !*((_BYTE *)v17 + v18 + 12) )
  {
    if ( v17[v18 + 4] )
    {
      ++v20;
      v19 = (unsigned __int64 *)v17[v18 + 4];
    }
    if ( ++v18 == 16 )
    {
      if ( v20 <= 1 )
      {
        *v4 = v19;
        sub_16856A0(v17);
      }
      return;
    }
  }
}
