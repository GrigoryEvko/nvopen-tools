// Function: sub_2EBF8D0
// Address: 0x2ebf8d0
//
unsigned __int16 *__fastcall sub_2EBF8D0(_QWORD *a1, unsigned int a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r15
  __int16 *v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdi
  __int16 v9; // r14
  __int16 *v10; // rbx
  __int64 v11; // rax
  unsigned __int16 *result; // rax
  __int64 v13; // rdx
  unsigned __int16 *v14; // rbx
  unsigned __int16 *v15; // r13
  unsigned __int16 *v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned __int16 *v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned __int16 *v22; // rdi
  unsigned __int16 *v23; // rdx
  int v24; // esi
  unsigned __int16 *v25; // r14

  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a1 + 16LL) + 200LL))(*(_QWORD *)(*a1 + 16LL));
  v4 = (_QWORD *)v3;
  if ( !*((_BYTE *)a1 + 176) )
  {
    v5 = (__int16 *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 72LL))(v3, *a1);
    v8 = (__int64)(a1 + 23);
    v9 = *v5;
    v10 = v5;
    v11 = a1[24];
    if ( v9 )
    {
      do
      {
        if ( (unsigned __int64)(v11 + 1) > a1[25] )
        {
          sub_C8D290(v8, a1 + 26, v11 + 1, 2u, v6, v7);
          v11 = a1[24];
        }
        ++v10;
        *(_WORD *)(a1[23] + 2 * v11) = v9;
        v11 = a1[24] + 1LL;
        a1[24] = v11;
        v9 = *v10;
      }
      while ( *v10 );
    }
    if ( (unsigned __int64)(v11 + 1) > a1[25] )
    {
      sub_C8D290(v8, a1 + 26, v11 + 1, 2u, v6, v7);
      v11 = a1[24];
    }
    *(_WORD *)(a1[23] + 2 * v11) = 0;
    ++a1[24];
    *((_BYTE *)a1 + 176) = 1;
  }
  result = (unsigned __int16 *)sub_E922F0(v4, a2);
  v14 = &result[v13];
  v15 = result;
  if ( result != v14 )
  {
    v16 = (unsigned __int16 *)a1[23];
    v17 = a1[24];
    while ( 1 )
    {
      result = (unsigned __int16 *)*v15;
      v18 = 2 * v17;
      v19 = &v16[v17];
      v20 = (2 * v17) >> 3;
      v21 = v18 >> 1;
      if ( v20 > 0 )
        break;
      v22 = v16;
LABEL_31:
      switch ( v21 )
      {
        case 2LL:
          goto LABEL_37;
        case 3LL:
          if ( (_DWORD)result != *v22 )
          {
            ++v22;
LABEL_37:
            if ( (_DWORD)result != *v22 )
            {
              ++v22;
LABEL_39:
              v25 = v19;
              if ( (_DWORD)result != *v22 )
                goto LABEL_25;
            }
          }
LABEL_18:
          if ( v19 != v22 )
          {
            v23 = v22 + 1;
            if ( v19 == v22 + 1 )
            {
              v25 = v22;
            }
            else
            {
              do
              {
                v24 = *v23;
                if ( (_DWORD)result != v24 )
                  *v22++ = v24;
                ++v23;
              }
              while ( v19 != v23 );
              v16 = (unsigned __int16 *)a1[23];
              result = &v16[a1[24]];
              v25 = (unsigned __int16 *)((char *)v22 + (char *)result - (char *)v19);
              if ( v19 != result )
              {
                result = (unsigned __int16 *)memmove(v22, v19, (char *)result - (char *)v19);
                v16 = (unsigned __int16 *)a1[23];
              }
            }
            goto LABEL_25;
          }
          break;
        case 1LL:
          goto LABEL_39;
      }
      v25 = v19;
LABEL_25:
      ++v15;
      v17 = v25 - v16;
      a1[24] = v17;
      if ( v14 == v15 )
        return result;
    }
    v22 = v16;
    while ( (_DWORD)result != *v22 )
    {
      if ( (_DWORD)result == v22[1] )
      {
        ++v22;
        goto LABEL_18;
      }
      if ( (_DWORD)result == v22[2] )
      {
        v22 += 2;
        goto LABEL_18;
      }
      if ( (_DWORD)result == v22[3] )
      {
        v22 += 3;
        goto LABEL_18;
      }
      v22 += 4;
      if ( &v16[4 * v20] == v22 )
      {
        v21 = v19 - v22;
        goto LABEL_31;
      }
    }
    goto LABEL_18;
  }
  return result;
}
