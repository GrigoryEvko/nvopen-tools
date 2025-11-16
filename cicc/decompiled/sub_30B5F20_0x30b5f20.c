// Function: sub_30B5F20
// Address: 0x30b5f20
//
__int64 __fastcall sub_30B5F20(char *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r10
  __int64 v5; // r15
  __int64 *v7; // r12
  __int64 v8; // r8
  __int64 v9; // rdx
  char *v10; // rdi
  __int64 v11; // rax
  __int16 v12; // si
  __int16 v13; // r11
  int v14; // r9d
  int v15; // ecx
  int v16; // ebx
  __int64 v17; // r11
  __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 *v20; // rdi
  __int16 v21; // r11
  __int64 *v22; // rbx
  __int64 *v23; // rax
  int v24; // ecx
  int v25; // r9d
  __int64 *v26; // r13
  int v27; // ecx
  int v28; // r11d
  __int64 v29; // rbx
  __int64 v30; // rbx
  __int64 v31; // r12
  __int64 v32; // rcx
  __int64 *v33; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 128 )
    return result;
  v4 = a2;
  v5 = a3;
  if ( !a3 )
  {
    v26 = a2;
    goto LABEL_40;
  }
  v7 = (__int64 *)(a1 + 8);
  v33 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    v8 = *((_QWORD *)a1 + 1);
    v9 = *(v4 - 1);
    --v5;
    v10 = &a1[8 * (result >> 4)];
    v11 = *(_QWORD *)v10;
    v12 = *(_WORD *)(v9 + 24);
    v13 = *(_WORD *)(*(_QWORD *)v10 + 24LL);
    if ( *(_WORD *)(v8 + 24) == 6 )
    {
      v14 = *(_DWORD *)(v8 + 40);
      v15 = 1;
      if ( v13 != 6 )
        goto LABEL_7;
    }
    else
    {
      v14 = 1;
      if ( v13 != 6 )
      {
        v15 = 1;
        if ( v12 != 6 )
        {
          v29 = *(_QWORD *)a1;
          goto LABEL_46;
        }
LABEL_31:
        v18 = *(_QWORD *)a1;
        v28 = *(_DWORD *)(v9 + 40);
        v29 = *(_QWORD *)a1;
        if ( v14 <= v28 )
          goto LABEL_35;
        goto LABEL_32;
      }
    }
    v15 = *(_DWORD *)(v11 + 40);
LABEL_7:
    if ( v15 < v14 )
    {
      v16 = 1;
      if ( v12 == 6 )
        v16 = *(_DWORD *)(v9 + 40);
      v17 = *(_QWORD *)a1;
      if ( v15 <= v16 )
      {
        if ( v16 >= v14 )
        {
          *(_QWORD *)a1 = v8;
          v18 = v17;
          *((_QWORD *)a1 + 1) = v17;
          v19 = *(v4 - 1);
        }
        else
        {
          *(_QWORD *)a1 = v9;
          v19 = v17;
          *(v4 - 1) = v17;
          v8 = *(_QWORD *)a1;
          v18 = *((_QWORD *)a1 + 1);
        }
      }
      else
      {
        *(_QWORD *)a1 = v11;
        *(_QWORD *)v10 = v17;
        v8 = *(_QWORD *)a1;
        v18 = *((_QWORD *)a1 + 1);
        v19 = *(v4 - 1);
      }
      goto LABEL_12;
    }
    v28 = 1;
    if ( v12 == 6 )
      goto LABEL_31;
    v18 = *(_QWORD *)a1;
    v29 = *(_QWORD *)a1;
    if ( v14 <= 1 )
    {
LABEL_35:
      if ( v15 > v28 )
      {
        *(_QWORD *)a1 = v9;
        v19 = v18;
        *(v4 - 1) = v18;
        v8 = *(_QWORD *)a1;
        v18 = *((_QWORD *)a1 + 1);
        goto LABEL_12;
      }
LABEL_46:
      *(_QWORD *)a1 = v11;
      *(_QWORD *)v10 = v29;
      v8 = *(_QWORD *)a1;
      v18 = *((_QWORD *)a1 + 1);
      v19 = *(v4 - 1);
      goto LABEL_12;
    }
LABEL_32:
    *(_QWORD *)a1 = v8;
    *((_QWORD *)a1 + 1) = v18;
    v19 = *(v4 - 1);
LABEL_12:
    v20 = v33;
    v21 = *(_WORD *)(v8 + 24);
    v22 = v7;
    v23 = v4;
    while ( 1 )
    {
      v26 = v22;
      if ( *(_WORD *)(v18 + 24) == 6 )
      {
        v24 = *(_DWORD *)(v18 + 40);
        v25 = 1;
        if ( v21 != 6 )
          goto LABEL_15;
      }
      else
      {
        v25 = 1;
        v24 = 1;
        if ( v21 != 6 )
          goto LABEL_19;
      }
      v25 = *(_DWORD *)(v8 + 40);
LABEL_15:
      if ( v25 < v24 )
        goto LABEL_16;
LABEL_19:
      for ( --v23; ; --v23 )
      {
        v27 = 1;
        if ( *(_WORD *)(v19 + 24) == 6 )
          v27 = *(_DWORD *)(v19 + 40);
        if ( v27 >= v25 )
          break;
        v19 = *(v23 - 1);
      }
      if ( v22 >= v23 )
        break;
      *v22 = v19;
      v19 = *(v23 - 1);
      *v23 = v18;
      v8 = *(_QWORD *)a1;
      v21 = *(_WORD *)(*(_QWORD *)a1 + 24LL);
LABEL_16:
      v18 = *v20;
      ++v22;
      ++v20;
    }
    sub_30B5F20(v22, v4, v5);
    result = (char *)v22 - a1;
    if ( (char *)v22 - a1 > 128 )
    {
      if ( v5 )
      {
        v4 = v22;
        continue;
      }
LABEL_40:
      v30 = result >> 3;
      v31 = ((result >> 3) - 2) >> 1;
      sub_30B5D80((__int64)a1, v31, result >> 3, *(_QWORD *)&a1[8 * v31]);
      do
      {
        --v31;
        sub_30B5D80((__int64)a1, v31, v30, *(_QWORD *)&a1[8 * v31]);
      }
      while ( v31 );
      do
      {
        v32 = *--v26;
        *v26 = *(_QWORD *)a1;
        result = sub_30B5D80((__int64)a1, 0, ((char *)v26 - a1) >> 3, v32);
      }
      while ( (char *)v26 - a1 > 8 );
    }
    return result;
  }
}
