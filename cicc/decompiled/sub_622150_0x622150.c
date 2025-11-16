// Function: sub_622150
// Address: 0x622150
//
__int64 __fastcall sub_622150(char *a1, __m128i *a2, int a3, _DWORD *a4)
{
  char *v5; // rdx
  unsigned __int8 v7; // cl
  char *v8; // rbx
  int v9; // r14d
  int v10; // r9d
  int v11; // esi
  unsigned __int8 *v12; // r8
  char v13; // cl
  __int64 v14; // rax
  __int64 result; // rax
  int v16; // eax
  _BYTE *v17; // rcx
  int v18; // edi
  int v19; // edx
  int v20; // eax
  int v21; // ebx
  int v23; // [rsp+14h] [rbp-BCh]
  int v24; // [rsp+18h] [rbp-B8h]
  _BOOL4 v25; // [rsp+3Ch] [rbp-94h] BYREF
  __m128i v26; // [rsp+40h] [rbp-90h] BYREF
  __int16 v27[8]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE v28[112]; // [rsp+60h] [rbp-70h] BYREF

  v5 = a1 + 1;
  v7 = *a1;
  v25 = 0;
  if ( v7 == 43 )
  {
    v23 = 0;
    v7 = *++a1;
    ++v5;
  }
  else
  {
    v23 = 0;
    if ( v7 == 45 )
    {
      v23 = 1;
      v7 = *++a1;
      ++v5;
    }
  }
  v8 = v28;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = v28;
  while ( 1 )
  {
    if ( (unsigned int)v7 - 48 <= 9 )
    {
      *v12 = v7;
      ++v11;
      ++v12;
      goto LABEL_6;
    }
    if ( v7 != 46 )
      break;
    v9 = v11;
    v10 = 1;
LABEL_6:
    v7 = *v5;
    a1 = v5++;
  }
  v24 = v10;
  v13 = v7 & 0xDF;
  *v12 = 0;
  if ( v10 )
  {
    if ( v13 != 69 )
      goto LABEL_12;
LABEL_25:
    v16 = *v5;
    if ( *v5 == 43 )
    {
      v16 = a1[2];
      v17 = a1 + 3;
      if ( a1[2] )
      {
        v18 = 0;
        goto LABEL_29;
      }
    }
    else if ( (_BYTE)v16 == 45 )
    {
      v16 = a1[2];
      v17 = a1 + 3;
      if ( a1[2] )
      {
        v18 = 1;
        goto LABEL_29;
      }
    }
    else
    {
      v17 = a1 + 2;
      if ( (_BYTE)v16 )
      {
        v18 = 0;
LABEL_29:
        v19 = 0;
        do
        {
          ++v17;
          v19 = v16 + 10 * v19 - 48;
          v16 = (char)*(v17 - 1);
        }
        while ( *(v17 - 1) );
        v24 = 0;
        v20 = v9 + v19;
        v9 -= v19;
        if ( !v18 )
          v9 = v20;
        if ( v11 <= v9 )
        {
LABEL_34:
          v24 = v9 - v11;
          goto LABEL_35;
        }
        goto LABEL_13;
      }
    }
LABEL_12:
    v24 = 0;
    if ( v11 <= v9 )
      goto LABEL_34;
    goto LABEL_13;
  }
  if ( v13 == 69 )
  {
    v9 = v11;
    goto LABEL_25;
  }
LABEL_35:
  v9 = v11;
LABEL_13:
  sub_620D80(a2, 0);
  v14 = (__int64)sub_620D80(&v26, 10);
  if ( v9 <= 0 )
  {
    if ( v25 )
      goto LABEL_21;
LABEL_37:
    if ( v24 )
    {
      v21 = 0;
      do
      {
        sub_621F20(a2, &v26, a3, &v25);
        LODWORD(v14) = v25;
        if ( v25 )
          goto LABEL_21;
      }
      while ( v24 != ++v21 );
    }
    LOBYTE(v14) = a3 == 0;
    result = v23 & (unsigned int)v14;
  }
  else
  {
    while ( 1 )
    {
      sub_620D80(v27, *v8 - 48LL);
      sub_621F20(a2, &v26, 1, &v25);
      if ( v25 )
        break;
      if ( a3 && (v23 & 1) != 0 )
      {
        v14 = sub_6215F0((unsigned __int16 *)a2, v27, a3, &v25);
        if ( v25 )
          break;
      }
      else
      {
        v14 = sub_621270((unsigned __int16 *)a2, v27, a3, &v25);
        if ( v25 )
          break;
      }
      if ( ++v8 == &v28[v9 - 1 + 1] )
        goto LABEL_37;
    }
LABEL_21:
    result = 1;
  }
  *a4 = result;
  return result;
}
