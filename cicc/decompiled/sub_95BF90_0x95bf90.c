// Function: sub_95BF90
// Address: 0x95bf90
//
__int64 __fastcall sub_95BF90(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rsi
  _BYTE *v5; // rdi
  __int64 v6; // rdx
  const void *v7; // rsi
  _BYTE *v8; // rdi
  __int64 v9; // rdx
  const void *v10; // rsi
  _BYTE *v11; // rdi
  __int64 v12; // rdx
  __int64 result; // rax
  size_t v14; // rdx
  size_t v15; // rdx
  size_t v16; // rdx

  v4 = *(_BYTE **)a2;
  v5 = *(_BYTE **)a1;
  if ( v4 == (_BYTE *)(a2 + 16) )
  {
    v14 = *(_QWORD *)(a2 + 8);
    if ( v14 )
    {
      if ( v14 == 1 )
        *v5 = *(_BYTE *)(a2 + 16);
      else
        memcpy(v5, v4, v14);
      v14 = *(_QWORD *)(a2 + 8);
      v5 = *(_BYTE **)a1;
    }
    *(_QWORD *)(a1 + 8) = v14;
    v5[v14] = 0;
    v5 = *(_BYTE **)a2;
  }
  else
  {
    if ( v5 == (_BYTE *)(a1 + 16) )
    {
      *(_QWORD *)a1 = v4;
      *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    }
    else
    {
      *(_QWORD *)a1 = v4;
      v6 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
      if ( v5 )
      {
        *(_QWORD *)a2 = v5;
        *(_QWORD *)(a2 + 16) = v6;
        goto LABEL_5;
      }
    }
    *(_QWORD *)a2 = a2 + 16;
    v5 = (_BYTE *)(a2 + 16);
  }
LABEL_5:
  *(_QWORD *)(a2 + 8) = 0;
  *v5 = 0;
  v7 = *(const void **)(a2 + 32);
  v8 = *(_BYTE **)(a1 + 32);
  if ( v7 == (const void *)(a2 + 48) )
  {
    v15 = *(_QWORD *)(a2 + 40);
    if ( v15 )
    {
      if ( v15 == 1 )
        *v8 = *(_BYTE *)(a2 + 48);
      else
        memcpy(v8, v7, v15);
      v15 = *(_QWORD *)(a2 + 40);
      v8 = *(_BYTE **)(a1 + 32);
    }
    *(_QWORD *)(a1 + 40) = v15;
    v8[v15] = 0;
    v8 = *(_BYTE **)(a2 + 32);
  }
  else
  {
    if ( v8 == (_BYTE *)(a1 + 48) )
    {
      *(_QWORD *)(a1 + 32) = v7;
      *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
      *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
    }
    else
    {
      *(_QWORD *)(a1 + 32) = v7;
      v9 = *(_QWORD *)(a1 + 48);
      *(_QWORD *)(a1 + 40) = *(_QWORD *)(a2 + 40);
      *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
      if ( v8 )
      {
        *(_QWORD *)(a2 + 32) = v8;
        *(_QWORD *)(a2 + 48) = v9;
        goto LABEL_9;
      }
    }
    *(_QWORD *)(a2 + 32) = a2 + 48;
    v8 = (_BYTE *)(a2 + 48);
  }
LABEL_9:
  *(_QWORD *)(a2 + 40) = 0;
  *v8 = 0;
  v10 = *(const void **)(a2 + 64);
  v11 = *(_BYTE **)(a1 + 64);
  if ( v10 != (const void *)(a2 + 80) )
  {
    if ( v11 == (_BYTE *)(a1 + 80) )
    {
      *(_QWORD *)(a1 + 64) = v10;
      *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
      *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
    }
    else
    {
      *(_QWORD *)(a1 + 64) = v10;
      v12 = *(_QWORD *)(a1 + 80);
      *(_QWORD *)(a1 + 72) = *(_QWORD *)(a2 + 72);
      *(_QWORD *)(a1 + 80) = *(_QWORD *)(a2 + 80);
      if ( v11 )
      {
        *(_QWORD *)(a2 + 64) = v11;
        *(_QWORD *)(a2 + 80) = v12;
        goto LABEL_13;
      }
    }
    *(_QWORD *)(a2 + 64) = a2 + 80;
    v11 = (_BYTE *)(a2 + 80);
    goto LABEL_13;
  }
  v16 = *(_QWORD *)(a2 + 72);
  if ( v16 )
  {
    if ( v16 == 1 )
      *v11 = *(_BYTE *)(a2 + 80);
    else
      memcpy(v11, v10, v16);
    v16 = *(_QWORD *)(a2 + 72);
    v11 = *(_BYTE **)(a1 + 64);
  }
  *(_QWORD *)(a1 + 72) = v16;
  v11[v16] = 0;
  v11 = *(_BYTE **)(a2 + 64);
LABEL_13:
  *(_QWORD *)(a2 + 72) = 0;
  *v11 = 0;
  result = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a1 + 96) = result;
  return result;
}
