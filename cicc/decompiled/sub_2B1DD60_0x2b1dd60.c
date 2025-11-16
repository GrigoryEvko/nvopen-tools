// Function: sub_2B1DD60
// Address: 0x2b1dd60
//
__int64 *__fastcall sub_2B1DD60(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 *v4; // r12
  __int64 v5; // rsi
  __int64 v6; // rax
  unsigned __int8 *v7; // r15
  __int64 *v8; // r13
  int v9; // eax
  unsigned __int8 *v10; // r15
  int v11; // eax
  unsigned __int8 *v12; // r15
  int v13; // eax
  __int64 v14; // r15
  int v15; // eax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  int v22; // eax
  __int64 v23; // rbx
  int v24; // eax
  __int64 v25; // rbx
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax

  v4 = a1;
  v5 = a2 - (_QWORD)a1;
  v6 = v5 >> 3;
  if ( v5 >> 5 > 0 )
  {
    do
    {
      v14 = *v4;
      v15 = *(unsigned __int8 *)*v4;
      if ( (_BYTE)v15 != 90 && (unsigned int)(v15 - 12) > 1 )
      {
        if ( !*a3 )
          return v4;
        if ( (unsigned __int8)sub_BD3660(*v4, 64) )
          return v4;
        v19 = *(_QWORD *)(v14 + 16);
        if ( !v19 )
          return v4;
        while ( **(_BYTE **)(v19 + 24) != 91 )
        {
          v19 = *(_QWORD *)(v19 + 8);
          if ( !v19 )
            return v4;
        }
      }
      v7 = (unsigned __int8 *)v4[1];
      v8 = v4 + 1;
      v9 = *v7;
      if ( (_BYTE)v9 != 90 && (unsigned int)(v9 - 12) > 1 )
      {
        if ( !*a3 )
          return v8;
        if ( (unsigned __int8)sub_BD3660(v4[1], 64) )
          return v8;
        v20 = *((_QWORD *)v7 + 2);
        if ( !v20 )
          return v8;
        while ( **(_BYTE **)(v20 + 24) != 91 )
        {
          v20 = *(_QWORD *)(v20 + 8);
          if ( !v20 )
            return v8;
        }
      }
      v10 = (unsigned __int8 *)v4[2];
      v8 = v4 + 2;
      v11 = *v10;
      if ( (_BYTE)v11 != 90 && (unsigned int)(v11 - 12) > 1 )
      {
        if ( !*a3 )
          return v8;
        if ( (unsigned __int8)sub_BD3660(v4[2], 64) )
          return v8;
        v17 = *((_QWORD *)v10 + 2);
        if ( !v17 )
          return v8;
        while ( **(_BYTE **)(v17 + 24) != 91 )
        {
          v17 = *(_QWORD *)(v17 + 8);
          if ( !v17 )
            return v8;
        }
      }
      v12 = (unsigned __int8 *)v4[3];
      v8 = v4 + 3;
      v13 = *v12;
      if ( (_BYTE)v13 != 90 && (unsigned int)(v13 - 12) > 1 )
      {
        if ( !*a3 )
          return v8;
        if ( (unsigned __int8)sub_BD3660(v4[3], 64) )
          return v8;
        v18 = *((_QWORD *)v12 + 2);
        if ( !v18 )
          return v8;
        while ( **(_BYTE **)(v18 + 24) != 91 )
        {
          v18 = *(_QWORD *)(v18 + 8);
          if ( !v18 )
            return v8;
        }
      }
      v4 += 4;
    }
    while ( &a1[4 * (v5 >> 5)] != v4 );
    v6 = (a2 - (__int64)v4) >> 3;
  }
  if ( v6 == 2 )
  {
LABEL_46:
    v23 = *v4;
    v24 = *(unsigned __int8 *)*v4;
    if ( (_BYTE)v24 != 90 && (unsigned int)(v24 - 12) > 1 )
    {
      if ( !*a3 )
        return v4;
      if ( (unsigned __int8)sub_BD3660(*v4, 64) )
        return v4;
      v28 = *(_QWORD *)(v23 + 16);
      if ( !v28 )
        return v4;
      while ( **(_BYTE **)(v28 + 24) != 91 )
      {
        v28 = *(_QWORD *)(v28 + 8);
        if ( !v28 )
          return v4;
      }
    }
    ++v4;
    goto LABEL_49;
  }
  if ( v6 == 3 )
  {
    v21 = *v4;
    v22 = *(unsigned __int8 *)*v4;
    if ( (_BYTE)v22 != 90 && (unsigned int)(v22 - 12) > 1 )
    {
      if ( !*a3 )
        return v4;
      if ( (unsigned __int8)sub_BD3660(*v4, 64) )
        return v4;
      v29 = *(_QWORD *)(v21 + 16);
      if ( !v29 )
        return v4;
      while ( **(_BYTE **)(v29 + 24) != 91 )
      {
        v29 = *(_QWORD *)(v29 + 8);
        if ( !v29 )
          return v4;
      }
    }
    ++v4;
    goto LABEL_46;
  }
  if ( v6 != 1 )
    return (__int64 *)a2;
LABEL_49:
  v25 = *v4;
  v26 = *(unsigned __int8 *)*v4;
  if ( (_BYTE)v26 == 90 || (unsigned int)(v26 - 12) <= 1 )
    return (__int64 *)a2;
  if ( *a3 )
  {
    if ( !(unsigned __int8)sub_BD3660(*v4, 64) )
    {
      v27 = *(_QWORD *)(v25 + 16);
      if ( v27 )
      {
        while ( **(_BYTE **)(v27 + 24) != 91 )
        {
          v27 = *(_QWORD *)(v27 + 8);
          if ( !v27 )
            return v4;
        }
        return (__int64 *)a2;
      }
    }
  }
  return v4;
}
