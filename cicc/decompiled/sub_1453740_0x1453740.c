// Function: sub_1453740
// Address: 0x1453740
//
__int64 __fastcall sub_1453740(char *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 *v4; // r10
  __int64 v5; // r15
  __int64 *v7; // r12
  __int64 v8; // r9
  __int64 v9; // rdx
  char *v10; // rdi
  __int64 v11; // rax
  __int16 v12; // si
  int v13; // r8d
  int v14; // ecx
  int v15; // esi
  __int64 v16; // r11
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 *v19; // rdi
  __int16 v20; // r11
  __int64 *v21; // rbx
  __int64 *v22; // rax
  int v23; // ecx
  int v24; // r8d
  __int64 *v25; // r13
  int v26; // ecx
  int v27; // r11d
  __int64 v28; // rbx
  __int64 v29; // rbx
  __int64 v30; // r12
  __int64 v31; // rcx
  __int64 *v32; // [rsp-40h] [rbp-40h]

  result = (char *)a2 - a1;
  if ( (char *)a2 - a1 <= 128 )
    return result;
  v4 = a2;
  v5 = a3;
  if ( !a3 )
  {
    v25 = a2;
    goto LABEL_40;
  }
  v7 = (__int64 *)(a1 + 8);
  v32 = (__int64 *)(a1 + 16);
  while ( 2 )
  {
    v8 = *((_QWORD *)a1 + 1);
    --v5;
    v9 = *(v4 - 1);
    v10 = &a1[8 * (result >> 4)];
    v11 = *(_QWORD *)v10;
    v12 = *(_WORD *)(*(_QWORD *)v10 + 24LL);
    if ( *(_WORD *)(v8 + 24) == 5 )
    {
      v13 = *(_DWORD *)(v8 + 40);
      v14 = 1;
      if ( v12 != 5 )
        goto LABEL_7;
    }
    else
    {
      v13 = 1;
      if ( v12 != 5 )
      {
        v14 = 1;
        if ( *(_WORD *)(v9 + 24) != 5 )
        {
          v28 = *(_QWORD *)a1;
          goto LABEL_46;
        }
LABEL_31:
        v17 = *(_QWORD *)a1;
        v27 = *(_DWORD *)(v9 + 40);
        v28 = *(_QWORD *)a1;
        if ( v13 <= v27 )
          goto LABEL_35;
        goto LABEL_32;
      }
    }
    v14 = *(_DWORD *)(v11 + 40);
LABEL_7:
    if ( v14 < v13 )
    {
      v15 = 1;
      if ( *(_WORD *)(v9 + 24) == 5 )
        v15 = *(_DWORD *)(v9 + 40);
      v16 = *(_QWORD *)a1;
      if ( v14 <= v15 )
      {
        if ( v15 >= v13 )
        {
          *(_QWORD *)a1 = v8;
          v17 = v16;
          *((_QWORD *)a1 + 1) = v16;
          v18 = *(v4 - 1);
        }
        else
        {
          *(_QWORD *)a1 = v9;
          v18 = v16;
          *(v4 - 1) = v16;
          v8 = *(_QWORD *)a1;
          v17 = *((_QWORD *)a1 + 1);
        }
      }
      else
      {
        *(_QWORD *)a1 = v11;
        *(_QWORD *)v10 = v16;
        v8 = *(_QWORD *)a1;
        v17 = *((_QWORD *)a1 + 1);
        v18 = *(v4 - 1);
      }
      goto LABEL_12;
    }
    v27 = 1;
    if ( *(_WORD *)(v9 + 24) == 5 )
      goto LABEL_31;
    v17 = *(_QWORD *)a1;
    v28 = *(_QWORD *)a1;
    if ( v13 <= 1 )
    {
LABEL_35:
      if ( v14 > v27 )
      {
        *(_QWORD *)a1 = v9;
        v18 = v17;
        *(v4 - 1) = v17;
        v8 = *(_QWORD *)a1;
        v17 = *((_QWORD *)a1 + 1);
        goto LABEL_12;
      }
LABEL_46:
      *(_QWORD *)a1 = v11;
      *(_QWORD *)v10 = v28;
      v8 = *(_QWORD *)a1;
      v17 = *((_QWORD *)a1 + 1);
      v18 = *(v4 - 1);
      goto LABEL_12;
    }
LABEL_32:
    *(_QWORD *)a1 = v8;
    *((_QWORD *)a1 + 1) = v17;
    v18 = *(v4 - 1);
LABEL_12:
    v19 = v32;
    v20 = *(_WORD *)(v8 + 24);
    v21 = v7;
    v22 = v4;
    while ( 1 )
    {
      v25 = v21;
      if ( *(_WORD *)(v17 + 24) == 5 )
      {
        v23 = *(_DWORD *)(v17 + 40);
        v24 = 1;
        if ( v20 != 5 )
          goto LABEL_15;
      }
      else
      {
        v23 = 1;
        v24 = 1;
        if ( v20 != 5 )
          goto LABEL_19;
      }
      v24 = *(_DWORD *)(v8 + 40);
LABEL_15:
      if ( v24 < v23 )
        goto LABEL_16;
LABEL_19:
      for ( --v22; ; --v22 )
      {
        v26 = 1;
        if ( *(_WORD *)(v18 + 24) == 5 )
          v26 = *(_DWORD *)(v18 + 40);
        if ( v26 >= v24 )
          break;
        v18 = *(v22 - 1);
      }
      if ( v21 >= v22 )
        break;
      *v21 = v18;
      v18 = *(v22 - 1);
      *v22 = v17;
      v8 = *(_QWORD *)a1;
      v20 = *(_WORD *)(*(_QWORD *)a1 + 24LL);
LABEL_16:
      v17 = *v19;
      ++v21;
      ++v19;
    }
    sub_1453740(v21, v4, v5);
    result = (char *)v21 - a1;
    if ( (char *)v21 - a1 > 128 )
    {
      if ( v5 )
      {
        v4 = v21;
        continue;
      }
LABEL_40:
      v29 = result >> 3;
      v30 = ((result >> 3) - 2) >> 1;
      sub_1452E00((__int64)a1, v30, result >> 3, *(_QWORD *)&a1[8 * v30]);
      do
      {
        --v30;
        sub_1452E00((__int64)a1, v30, v29, *(_QWORD *)&a1[8 * v30]);
      }
      while ( v30 );
      do
      {
        v31 = *--v25;
        *v25 = *(_QWORD *)a1;
        result = sub_1452E00((__int64)a1, 0, ((char *)v25 - a1) >> 3, v31);
      }
      while ( (char *)v25 - a1 > 8 );
    }
    return result;
  }
}
