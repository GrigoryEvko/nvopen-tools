// Function: sub_130FEB0
// Address: 0x130feb0
//
void __fastcall sub_130FEB0(_QWORD *a1, __int64 a2, __int64 a3, _QWORD **a4)
{
  __int64 v5; // r13
  unsigned __int64 v6; // r12
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rsi
  char *v10; // rax
  __int64 v11; // r8
  unsigned __int64 v12; // rcx
  _QWORD **v13; // rcx
  _QWORD **v14; // r13
  _QWORD *v15; // rax
  const char *v16; // rax
  __int64 v17; // rdx
  _QWORD *v18; // rdx
  unsigned int i; // r9d
  __int64 v20; // r10
  _QWORD *v21; // r9
  _QWORD *v22; // r10

  if ( a3 )
  {
    v5 = 8 * a3;
    v6 = 0;
    do
    {
      v8 = *(_QWORD *)(*(_QWORD *)(a2 + 8) + v6);
      v9 = v8 & 0xFFFFFFFFC0000000LL;
      v10 = (char *)a1 + ((v8 >> 26) & 0xF0);
      v11 = *((_QWORD *)v10 + 54);
      if ( (v8 & 0xFFFFFFFFC0000000LL) == v11 )
      {
        v12 = *((_QWORD *)v10 + 55) + ((v8 >> 9) & 0x1FFFF8);
      }
      else if ( v9 == a1[86] )
      {
        a1[86] = v11;
        v17 = a1[87];
        a1[87] = *((_QWORD *)v10 + 55);
LABEL_11:
        *((_QWORD *)v10 + 54) = v9;
        *((_QWORD *)v10 + 55) = v17;
        v12 = v17 + ((v8 >> 9) & 0x1FFFF8);
      }
      else
      {
        v18 = a1 + 88;
        for ( i = 1; i != 8; ++i )
        {
          if ( v9 == *v18 )
          {
            v20 = 2LL * i;
            v21 = &a1[2 * i - 2];
            v22 = &a1[v20];
            v17 = v22[87];
            v22[86] = v21[86];
            v22[87] = v21[87];
            v21[86] = v11;
            v21[87] = *((_QWORD *)v10 + 55);
            goto LABEL_11;
          }
          v18 += 2;
        }
        v12 = sub_130D370((__int64)a1, (__int64)&unk_5060AE0, a1 + 54, v8, 1, 0);
      }
      a4[v6 / 8] = (_QWORD *)v12;
      v6 += 8LL;
    }
    while ( v5 != v6 );
    v13 = a4;
    v14 = &a4[(unsigned __int64)v5 / 8];
    do
    {
      v15 = *v13++;
      v16 = (const char *)(((__int64)(*v15 << 16) >> 16) & 0xFFFFFFFFFFFFFF80LL);
      *(v13 - 1) = v16;
      _mm_prefetch(v16, 1);
      _mm_prefetch(v16 + 64, 1);
    }
    while ( v14 != v13 );
  }
}
