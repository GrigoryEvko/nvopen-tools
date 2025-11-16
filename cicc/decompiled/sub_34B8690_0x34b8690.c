// Function: sub_34B8690
// Address: 0x34b8690
//
unsigned __int8 *__fastcall sub_34B8690(unsigned __int8 *a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  unsigned __int8 v7; // dl
  unsigned __int8 *v8; // r14
  __int64 v10; // r12
  __int64 *v11; // rcx
  int v12; // eax
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // r10
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 (*v22)(); // rax
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rsi
  _DWORD *v26; // rax
  __int64 v27; // r10
  _DWORD *v28; // rcx
  _DWORD *v29; // rdx
  unsigned __int64 v30; // rax
  _DWORD *v31; // rax
  _DWORD *i; // rdx
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+20h] [rbp-50h]

  v5 = (__int64)a1;
  v7 = *a1;
  if ( *a1 > 0x1Cu )
  {
    v8 = a1;
    v10 = a5;
    while ( 1 )
    {
      if ( (*((_DWORD *)v8 + 1) & 0x7FFFFFF) == 0 )
        return v8;
      if ( (v8[7] & 0x40) != 0 )
        v11 = (__int64 *)*((_QWORD *)v8 - 1);
      else
        v11 = (__int64 *)&v8[-32 * (*((_DWORD *)v8 + 1) & 0x7FFFFFF)];
      v5 = *v11;
      if ( v7 == 78 )
        goto LABEL_14;
      if ( v7 == 63 )
        break;
      if ( v7 == 77 )
      {
        if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v8 + 1) + 8LL) - 17 <= 1 )
          return v8;
        a5 = (__int64)sub_AE2980(v10, 0);
        if ( *(_DWORD *)(a5 + 4) != *(_DWORD *)(*(_QWORD *)(v5 + 8) + 8LL) >> 8 )
          return v8;
      }
      else
      {
        if ( v7 == 76 )
        {
          if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v8 + 1) + 8LL) - 17 <= 1 )
            return v8;
          a5 = (__int64)sub_AE2980(v10, 0);
          if ( *(_DWORD *)(a5 + 4) != *(_DWORD *)(*((_QWORD *)v8 + 1) + 8LL) >> 8 )
            return v8;
          goto LABEL_22;
        }
        if ( v7 == 67 )
        {
          v22 = *(__int64 (**)())(*(_QWORD *)a4 + 1384LL);
          if ( v22 != sub_2FE3470 )
          {
            if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))v22)(
                   a4,
                   *(_QWORD *)(v5 + 8),
                   *((_QWORD *)v8 + 1)) )
            {
              v23 = sub_BCAE30(*((_QWORD *)v8 + 1));
              if ( v23 > (unsigned int)*a3 )
                LODWORD(v23) = *a3;
              *a3 = v23;
              goto LABEL_15;
            }
          }
        }
        v12 = *v8;
        v13 = (unsigned int)(v12 - 34);
        if ( (unsigned __int8)(v12 - 34) <= 0x33u )
        {
          v14 = 0x8000000000041LL;
          if ( !_bittest64(&v14, v13) )
            return v8;
          v5 = sub_B494D0((__int64)v8, 52);
          if ( !v5 )
            return v8;
LABEL_14:
          if ( !sub_34B85F0(*(_QWORD *)(v5 + 8), *((_QWORD *)v8 + 1), a4, (__int64)v11, a5) )
            return v8;
          goto LABEL_15;
        }
        if ( (_BYTE)v12 != 94 )
        {
          if ( (_BYTE)v12 != 93 )
            return v8;
          v16 = *((unsigned int *)v8 + 20);
          v17 = *((_QWORD *)v8 + 9);
          v18 = *(unsigned int *)(a2 + 8);
          v19 = 4 * v16;
          v20 = v16;
          if ( v16 + v18 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
          {
            v33 = *((unsigned int *)v8 + 20);
            v34 = 4 * v16;
            v35 = *((_QWORD *)v8 + 9);
            sub_C8D5F0(a2, (const void *)(a2 + 16), v16 + v18, 4u, v16, v17);
            v16 = v33;
            v19 = v34;
            v17 = v35;
            v18 = *(unsigned int *)(a2 + 8);
            v20 = v33;
          }
          v21 = *(_QWORD *)a2 + 4 * v18;
          if ( v19 )
          {
            do
            {
              v21 += 4;
              *(_DWORD *)(v21 - 4) = *(_DWORD *)(v17 + v19 - 4 * v16 + 4 * v20-- - 4);
            }
            while ( v20 );
            v18 = *(unsigned int *)(a2 + 8);
          }
          a5 = v18 + v16;
          *(_DWORD *)(a2 + 8) = a5;
          goto LABEL_22;
        }
        v24 = *((unsigned int *)v8 + 20);
        v25 = *(unsigned int *)(a2 + 8);
        a5 = v25;
        if ( v24 > v25 )
          goto LABEL_22;
        v26 = (_DWORD *)*((_QWORD *)v8 + 9);
        v27 = *(_QWORD *)a2;
        v28 = &v26[v24];
        if ( v26 != v28 )
        {
          v29 = (_DWORD *)(v27 + 4 * v25 - 4);
          while ( *v26 == *v29 )
          {
            ++v26;
            --v29;
            if ( v28 == v26 )
              goto LABEL_47;
          }
          goto LABEL_22;
        }
LABEL_47:
        v30 = v25 - v24;
        if ( v25 != v25 - v24 )
        {
          a5 = (unsigned int)(v25 - v24);
          if ( v25 <= v30 )
          {
            if ( v30 > *(unsigned int *)(a2 + 12) )
            {
              sub_C8D5F0(a2, (const void *)(a2 + 16), v30, 4u, a5, v24);
              v30 = v25 - v24;
              v27 = *(_QWORD *)a2;
            }
            v31 = (_DWORD *)(v27 + 4 * v30);
            for ( i = (_DWORD *)(v27 + 4LL * *(unsigned int *)(a2 + 8)); v31 != i; ++i )
            {
              if ( i )
                *i = 0;
            }
          }
          *(_DWORD *)(a2 + 8) = v25 - v24;
        }
        v5 = *((_QWORD *)v8 - 4);
        if ( !v5 )
          return v8;
      }
LABEL_15:
      v7 = *(_BYTE *)v5;
      if ( *(_BYTE *)v5 <= 0x1Cu )
        return (unsigned __int8 *)v5;
      v8 = (unsigned __int8 *)v5;
    }
    if ( !(unsigned __int8)sub_B4DCF0((__int64)v8) )
      return v8;
LABEL_22:
    if ( !v5 )
      return v8;
    goto LABEL_15;
  }
  return (unsigned __int8 *)v5;
}
