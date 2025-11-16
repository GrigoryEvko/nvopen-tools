// Function: sub_D338A0
// Address: 0xd338a0
//
__int64 *__fastcall sub_D338A0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r13
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rsi
  _QWORD *v11; // rax
  _QWORD *v12; // rcx
  _QWORD *v13; // rdx
  __int64 v15; // rsi
  __int64 *v16; // r15
  __int64 v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // rdx
  __int64 v20; // rsi
  _QWORD *v21; // rax
  _QWORD *v22; // rdx
  __int64 v23; // rdi
  char v24; // al
  __int64 v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // rsi
  char v29; // cl
  _QWORD *v30; // rax
  _QWORD *v31; // rdx
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  __int64 v34; // rsi

  v4 = a1;
  v6 = (a2 - (__int64)a1) >> 5;
  v7 = (a2 - (__int64)a1) >> 3;
  if ( v6 <= 0 )
  {
LABEL_37:
    switch ( v7 )
    {
      case 2LL:
        v23 = *(_QWORD *)(a3 + 1112);
        v29 = *(_BYTE *)(v23 + 28);
        break;
      case 3LL:
        v23 = *(_QWORD *)(a3 + 1112);
        v28 = *v4;
        v29 = *(_BYTE *)(v23 + 28);
        if ( v29 )
        {
          v30 = *(_QWORD **)(v23 + 8);
          v31 = &v30[*(unsigned int *)(v23 + 20)];
          if ( v30 != v31 )
          {
            while ( v28 != *v30 )
            {
              if ( v31 == ++v30 )
                goto LABEL_63;
            }
            return v4;
          }
          v34 = v4[1];
          ++v4;
          goto LABEL_56;
        }
        if ( sub_C8CA60(v23, v28) )
          return v4;
        v23 = *(_QWORD *)(a3 + 1112);
        v29 = *(_BYTE *)(v23 + 28);
LABEL_63:
        ++v4;
        break;
      case 1LL:
        v23 = *(_QWORD *)(a3 + 1112);
        v24 = *(_BYTE *)(v23 + 28);
        goto LABEL_42;
      default:
        return (__int64 *)a2;
    }
    v34 = *v4;
    if ( !v29 )
    {
      if ( sub_C8CA60(v23, v34) )
        return v4;
      v23 = *(_QWORD *)(a3 + 1112);
      v24 = *(_BYTE *)(v23 + 28);
LABEL_69:
      ++v4;
LABEL_42:
      v25 = *v4;
      if ( !v24 )
      {
        if ( !sub_C8CA60(v23, v25) )
          return (__int64 *)a2;
        return v4;
      }
LABEL_43:
      v26 = *(_QWORD **)(v23 + 8);
      v27 = &v26[*(unsigned int *)(v23 + 20)];
      if ( v26 != v27 )
      {
        while ( v25 != *v26 )
        {
          if ( v27 == ++v26 )
            return (__int64 *)a2;
        }
        return v4;
      }
      return (__int64 *)a2;
    }
LABEL_56:
    v32 = *(_QWORD **)(v23 + 8);
    v33 = &v32[*(unsigned int *)(v23 + 20)];
    if ( v32 != v33 )
    {
      while ( v34 != *v32 )
      {
        if ( v33 == ++v32 )
        {
          v24 = 1;
          goto LABEL_69;
        }
      }
      return v4;
    }
    v25 = v4[1];
    ++v4;
    goto LABEL_43;
  }
  v8 = &a1[4 * v6];
  while ( 1 )
  {
    v9 = *(_QWORD *)(a3 + 1112);
    v10 = *v4;
    if ( *(_BYTE *)(v9 + 28) )
      break;
    if ( sub_C8CA60(v9, v10) )
      return v4;
    v9 = *(_QWORD *)(a3 + 1112);
    v15 = v4[1];
    v16 = v4 + 1;
    if ( *(_BYTE *)(v9 + 28) )
    {
      v11 = *(_QWORD **)(v9 + 8);
      v12 = &v11[*(unsigned int *)(v9 + 20)];
      goto LABEL_10;
    }
    if ( sub_C8CA60(v9, v15) )
      return v16;
    v9 = *(_QWORD *)(a3 + 1112);
    v17 = v4[2];
    v16 = v4 + 2;
    if ( *(_BYTE *)(v9 + 28) )
      goto LABEL_16;
    if ( sub_C8CA60(v9, v17) )
      return v16;
    v9 = *(_QWORD *)(a3 + 1112);
    v20 = v4[3];
    v16 = v4 + 3;
    if ( !*(_BYTE *)(v9 + 28) )
    {
      if ( sub_C8CA60(v9, v20) )
        return v16;
      goto LABEL_35;
    }
LABEL_25:
    v21 = *(_QWORD **)(v9 + 8);
    v22 = &v21[*(unsigned int *)(v9 + 20)];
    if ( v21 != v22 )
    {
      while ( *v21 != v20 )
      {
        if ( v22 == ++v21 )
          goto LABEL_35;
      }
      return v16;
    }
LABEL_35:
    v4 += 4;
    if ( v8 == v4 )
    {
      v7 = (a2 - (__int64)v4) >> 3;
      goto LABEL_37;
    }
  }
  v11 = *(_QWORD **)(v9 + 8);
  v12 = &v11[*(unsigned int *)(v9 + 20)];
  if ( v11 == v12 )
  {
LABEL_9:
    v15 = v4[1];
    v16 = v4 + 1;
LABEL_10:
    if ( v12 != v11 )
    {
      while ( *v11 != v15 )
      {
        if ( ++v11 == v12 )
          goto LABEL_15;
      }
      return v16;
    }
LABEL_15:
    v17 = v4[2];
    v16 = v4 + 2;
LABEL_16:
    v18 = *(_QWORD **)(v9 + 8);
    v19 = &v18[*(unsigned int *)(v9 + 20)];
    if ( v18 != v19 )
    {
      while ( *v18 != v17 )
      {
        if ( v19 == ++v18 )
          goto LABEL_24;
      }
      return v16;
    }
LABEL_24:
    v20 = v4[3];
    v16 = v4 + 3;
    goto LABEL_25;
  }
  v13 = *(_QWORD **)(v9 + 8);
  while ( v10 != *v13 )
  {
    if ( v12 == ++v13 )
      goto LABEL_9;
  }
  return v4;
}
