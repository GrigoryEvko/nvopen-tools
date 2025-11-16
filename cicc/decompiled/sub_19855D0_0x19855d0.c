// Function: sub_19855D0
// Address: 0x19855d0
//
void __fastcall sub_19855D0(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 i; // r15
  _QWORD *v11; // rax
  _QWORD *v12; // rdi
  _QWORD *v13; // r13
  _QWORD *v14; // rsi
  __int64 *v15; // rsi
  unsigned int v16; // edi
  __int64 *v17; // rcx
  _QWORD *v18; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v19; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v20; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v21; // [rsp+30h] [rbp-D0h]
  _BYTE v22[184]; // [rsp+48h] [rbp-B8h] BYREF

  if ( (unsigned __int8)sub_1648D00(a2, 33) )
    return;
  if ( *(_QWORD *)(a1 + 64) != a2 )
  {
    sub_16CCCB0(&v19, (__int64)v22, a3);
    v8 = sub_146F1B0(*(_QWORD *)(a1 + 16), a2);
    if ( *(_WORD *)(v8 + 24) == 7
      && *(_QWORD *)(v8 + 48) == *(_QWORD *)(a1 + 8)
      && (unsigned __int8)sub_1984A30(a1, a2, (__int64)&v19, a4, a5) )
    {
      if ( v21 != v20 )
        _libc_free(v21);
      return;
    }
    if ( v21 != v20 )
      _libc_free(v21);
  }
  v9 = *(__int64 **)(a3 + 8);
  if ( *(__int64 **)(a3 + 16) != v9 )
    goto LABEL_11;
  v15 = &v9[*(unsigned int *)(a3 + 28)];
  v16 = *(_DWORD *)(a3 + 28);
  if ( v9 != v15 )
  {
    v17 = 0;
    while ( a2 != *v9 )
    {
      if ( *v9 == -2 )
        v17 = v9;
      if ( v15 == ++v9 )
      {
        if ( !v17 )
          goto LABEL_28;
        *v17 = a2;
        --*(_DWORD *)(a3 + 32);
        ++*(_QWORD *)a3;
        goto LABEL_12;
      }
    }
    goto LABEL_12;
  }
LABEL_28:
  if ( v16 < *(_DWORD *)(a3 + 24) )
  {
    *(_DWORD *)(a3 + 28) = v16 + 1;
    *v15 = a2;
    ++*(_QWORD *)a3;
  }
  else
  {
LABEL_11:
    sub_16CCBA0(a3, a2);
  }
LABEL_12:
  for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v11 = sub_1648700(i);
    v12 = *(_QWORD **)(a1 + 5224);
    v18 = v11;
    v13 = v11;
    v14 = &v12[*(unsigned int *)(a1 + 5232)];
    if ( v14 == sub_1983C40(v12, (__int64)v14, (__int64 *)&v18) && *((_BYTE *)v13 + 16) > 0x17u )
    {
      switch ( *((_BYTE *)v13 + 16) )
      {
        case '#':
        case '%':
        case '\'':
        case '/':
        case '0':
        case '1':
        case '8':
        case '<':
        case '=':
        case '>':
          sub_16CCCB0(&v19, (__int64)v22, a3);
          sub_19855D0(a1, v13, &v19);
          if ( v21 != v20 )
            _libc_free(v21);
          break;
        default:
          continue;
      }
    }
  }
}
