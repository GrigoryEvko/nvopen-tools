// Function: sub_1BC14B0
// Address: 0x1bc14b0
//
void __fastcall sub_1BC14B0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r8
  unsigned __int64 *v4; // r15
  unsigned __int64 v5; // rax
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // r12
  __int64 v8; // rax
  unsigned __int64 *v9; // rbx
  __int64 v10; // r15
  _QWORD *i; // r12
  unsigned __int64 v12; // rax
  _QWORD *v13; // r12
  unsigned __int64 *v14; // rbx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rcx
  unsigned __int64 *v17; // rcx
  unsigned __int64 *v18; // rbx
  __int64 v19; // rax
  _QWORD *v20; // r12
  unsigned __int64 *v21; // rbx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // [rsp-48h] [rbp-48h]
  unsigned __int64 v25; // [rsp-48h] [rbp-48h]
  unsigned __int64 v26; // [rsp-48h] [rbp-48h]
  unsigned int v27; // [rsp-3Ch] [rbp-3Ch]

  if ( a1 != a2 )
  {
    v4 = *(unsigned __int64 **)a1;
    v5 = *(unsigned int *)(a1 + 8);
    v27 = *(_DWORD *)(a2 + 8);
    v3 = v27;
    v6 = *(unsigned __int64 **)a1;
    if ( v27 <= v5 )
    {
      v17 = *(unsigned __int64 **)a1;
      if ( v27 )
      {
        v20 = *(_QWORD **)a2;
        v21 = &v4[3 * v27];
        do
        {
          v22 = v4[2];
          v23 = v20[2];
          if ( v22 != v23 )
          {
            if ( v22 != -8 && v22 != 0 && v22 != -16 )
            {
              sub_1649B30(v4);
              v23 = v20[2];
            }
            v4[2] = v23;
            if ( v23 != 0 && v23 != -8 && v23 != -16 )
              sub_1649AC0(v4, *v20 & 0xFFFFFFFFFFFFFFF8LL);
          }
          v4 += 3;
          v20 += 3;
        }
        while ( v4 != v21 );
        v17 = *(unsigned __int64 **)a1;
        v5 = *(unsigned int *)(a1 + 8);
      }
      v18 = &v17[3 * v5];
      while ( v4 != v18 )
      {
        v19 = *(v18 - 1);
        v18 -= 3;
        if ( v19 != 0 && v19 != -8 && v19 != -16 )
          sub_1649B30(v18);
      }
    }
    else
    {
      if ( v27 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        if ( *(_DWORD *)(a1 + 8) )
        {
          v13 = *(_QWORD **)a2;
          v5 *= 24LL;
          v14 = (unsigned __int64 *)((char *)v4 + v5);
          do
          {
            v15 = v4[2];
            v16 = v13[2];
            if ( v15 != v16 )
            {
              if ( v15 != -8 && v15 != 0 && v15 != -16 )
              {
                v25 = v5;
                sub_1649B30(v4);
                v16 = v13[2];
                v5 = v25;
              }
              v4[2] = v16;
              if ( v16 != 0 && v16 != -8 && v16 != -16 )
              {
                v26 = v5;
                sub_1649AC0(v4, *v13 & 0xFFFFFFFFFFFFFFF8LL);
                v5 = v26;
              }
            }
            v4 += 3;
            v13 += 3;
          }
          while ( v4 != v14 );
          v3 = *(unsigned int *)(a2 + 8);
          v6 = *(unsigned __int64 **)a1;
        }
      }
      else
      {
        v7 = &v4[3 * v5];
        while ( v7 != v4 )
        {
          while ( 1 )
          {
            v8 = *(v7 - 1);
            v7 -= 3;
            if ( v8 == -8 || v8 == 0 || v8 == -16 )
              break;
            v24 = v3;
            sub_1649B30(v7);
            v3 = v24;
            if ( v7 == v4 )
              goto LABEL_9;
          }
        }
LABEL_9:
        *(_DWORD *)(a1 + 8) = 0;
        sub_170B450(a1, v3);
        v3 = *(unsigned int *)(a2 + 8);
        v6 = *(unsigned __int64 **)a1;
        v5 = 0;
      }
      v9 = (unsigned __int64 *)((char *)v6 + v5);
      v10 = *(_QWORD *)a2 + 24 * v3;
      for ( i = (_QWORD *)(v5 + *(_QWORD *)a2); (_QWORD *)v10 != i; v9 += 3 )
      {
        if ( v9 )
        {
          *v9 = 6;
          v9[1] = 0;
          v12 = i[2];
          v9[2] = v12;
          if ( v12 != -8 && v12 != 0 && v12 != -16 )
            sub_1649AC0(v9, *i & 0xFFFFFFFFFFFFFFF8LL);
        }
        i += 3;
      }
    }
    *(_DWORD *)(a1 + 8) = v27;
  }
}
