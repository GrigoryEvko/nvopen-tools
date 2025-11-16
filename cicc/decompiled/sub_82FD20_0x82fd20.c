// Function: sub_82FD20
// Address: 0x82fd20
//
void __fastcall sub_82FD20(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6, __int64 a7)
{
  __int64 v8; // rcx
  __int64 v9; // rbx
  char v10; // al
  __int64 v11; // r15
  char v12; // dl
  __int64 v13; // rax
  __int64 j; // r14
  __int64 v15; // rax
  bool v16; // dl
  __int64 v17; // rax
  __int64 v18; // rax
  char v19; // cl
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rsi
  char v24; // dl
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 i; // rax
  __int64 v32; // rax
  __int64 v34; // r14
  _DWORD *v35; // r8
  int v36; // eax
  __int64 v37; // [rsp+0h] [rbp-70h]
  bool v40; // [rsp+1Ch] [rbp-54h]
  bool v41; // [rsp+1Ch] [rbp-54h]
  _DWORD v42[20]; // [rsp+20h] [rbp-50h] BYREF

  v8 = a3;
  v9 = a3;
  v10 = *(_BYTE *)(a3 + 80);
  if ( v10 == 16 )
  {
    v8 = **(_QWORD **)(a3 + 88);
    v10 = *(_BYTE *)(v8 + 80);
  }
  if ( v10 == 24 )
    v8 = *(_QWORD *)(v8 + 88);
  if ( !a1[1].m128i_i8[0] )
    return;
  v11 = a1->m128i_i64[0];
  v12 = *(_BYTE *)(a1->m128i_i64[0] + 140);
  if ( v12 == 12 )
  {
    v13 = a1->m128i_i64[0];
    do
    {
      v13 = *(_QWORD *)(v13 + 160);
      v12 = *(_BYTE *)(v13 + 140);
    }
    while ( v12 == 12 );
  }
  if ( !v12 )
    return;
  j = *(_QWORD *)(a4 + 64);
  v37 = 0;
  if ( (unsigned __int8)(*(_BYTE *)(v8 + 80) - 10) <= 1u )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)(v8 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v32 = **(_QWORD **)(i + 168);
    v37 = v32;
    if ( v32 )
    {
      if ( (*(_BYTE *)(v32 + 35) & 1) != 0 )
      {
        v37 = *(_QWORD *)(v32 + 8);
        a2 = (unsigned int)a2;
        if ( (unsigned int)sub_8D32E0(v37) )
        {
          a2 = (unsigned int)a2;
          for ( j = sub_8D46C0(v37); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          v37 = j;
          v11 = a1->m128i_i64[0];
        }
        else
        {
          v11 = a1->m128i_i64[0];
        }
      }
      else
      {
        v37 = 0;
      }
    }
  }
  if ( (_DWORD)a2 )
  {
    if ( (unsigned int)sub_8DD3B0(v11) )
      v11 = dword_4D03B80;
    else
      v11 = sub_8D46C0(v11);
  }
  while ( *(_BYTE *)(v11 + 140) == 12 )
    v11 = *(_QWORD *)(v11 + 160);
  if ( dword_4F04C44 == -1
    && (v15 = qword_4F04C68[0] + 776LL * dword_4F04C64, (*(_BYTE *)(v15 + 6) & 6) == 0)
    && *(_BYTE *)(v15 + 4) != 12
    || !(unsigned int)sub_8D3D40(v11) && (*(_BYTE *)(v11 + 177) & 0x20) == 0 && (*(_BYTE *)(j + 177) & 0x20) == 0 )
  {
    v16 = a5 == 0;
    if ( a5 != 0 || a6 == 0 )
      goto LABEL_19;
    v36 = sub_6E6010();
    v16 = a5 == 0 && a6 != 0;
    if ( !v36 )
      goto LABEL_19;
    goto LABEL_88;
  }
  if ( *(_BYTE *)(a4 + 80) == 16 )
    goto LABEL_65;
  if ( j != v11 )
  {
    if ( j )
    {
      if ( dword_4F07588 )
      {
        v30 = *(_QWORD *)(v11 + 32);
        if ( *(_QWORD *)(j + 32) == v30 )
        {
          if ( v30 )
          {
            v16 = a5 == 0;
            if ( a5 != 0 || a6 == 0 )
            {
LABEL_55:
              if ( *(_QWORD *)(j + 32) == v30 && v30 )
              {
                j = v11;
                goto LABEL_24;
              }
              goto LABEL_22;
            }
            if ( !(unsigned int)sub_6E6010() )
            {
              v16 = a5 == 0 && a6 != 0;
LABEL_21:
              if ( dword_4F07588 )
              {
                v30 = *(_QWORD *)(v11 + 32);
                goto LABEL_55;
              }
LABEL_22:
              v40 = v16;
              v17 = sub_8D5CE0(v11, j);
              if ( v17 )
              {
                sub_6F7270(a1, v17, 0, v40, 0, 1, 0, 1);
              }
              else if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
              {
                sub_6E50A0();
              }
              goto LABEL_24;
            }
            goto LABEL_88;
          }
        }
      }
    }
    if ( !(unsigned int)sub_8D3D40(v11) )
    {
      a2 = j;
      if ( sub_8D5CE0(v11, j) )
      {
        v16 = a5 == 0;
        if ( a5 != 0 || a6 == 0 )
          goto LABEL_20;
        v41 = a5 == 0 && a6 != 0;
        if ( !(unsigned int)sub_6E6010() )
        {
          v16 = v41;
          goto LABEL_20;
        }
        goto LABEL_88;
      }
    }
LABEL_65:
    sub_6F40C0((__int64)a1, a2, v26, v27, v28, v29);
    return;
  }
  if ( !a6 || a5 )
  {
    j = v11;
    goto LABEL_25;
  }
  if ( (unsigned int)sub_6E6010() )
  {
LABEL_88:
    v42[0] = 0;
    v35 = 0;
    if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
      v35 = v42;
    sub_8843A0(v9, a4, a7, v11, v35);
    if ( v42[0] )
      sub_6E50A0();
    v16 = 1;
LABEL_19:
    if ( j == v11 )
      goto LABEL_24;
LABEL_20:
    if ( !j )
      goto LABEL_22;
    goto LABEL_21;
  }
LABEL_24:
  if ( *(_BYTE *)(a4 + 80) == 16 )
  {
    if ( v37 )
      return;
    v34 = *(_QWORD *)(*(_QWORD *)(a4 + 88) + 8LL);
    sub_6F7270(a1, v34, 0, 0, 0, 1, 1, 1);
    j = *(_QWORD *)(v34 + 40);
  }
LABEL_25:
  if ( a4 != v9 && !v37 )
  {
    v18 = *(_QWORD *)(v9 + 64);
    v19 = *(_BYTE *)(v9 + 80);
    if ( v18 == j || v18 && j && dword_4F07588 && (v20 = *(_QWORD *)(v18 + 32), *(_QWORD *)(j + 32) == v20) && v20 )
    {
LABEL_76:
      if ( v19 == 16 )
        sub_6F7270(a1, *(_QWORD *)(*(_QWORD *)(v9 + 88) + 8LL), 0, 0, 0, 1, 1, 1);
      return;
    }
    if ( v19 == 16 )
    {
      v21 = *(__int64 **)(v9 + 88);
      v9 = *v21;
      v19 = *(_BYTE *)(*v21 + 80);
    }
    if ( v19 == 24 )
      v9 = *(_QWORD *)(v9 + 88);
    v22 = sub_82C1B0(a4, 0, 0, (__int64)v42);
    if ( v22 )
    {
      while ( 1 )
      {
        v19 = *(_BYTE *)(v22 + 80);
        v23 = v22;
        v24 = v19;
        if ( v19 == 16 )
        {
          v23 = **(_QWORD **)(v22 + 88);
          v24 = *(_BYTE *)(v23 + 80);
        }
        if ( v24 == 24 )
        {
          v23 = *(_QWORD *)(v23 + 88);
          v24 = *(_BYTE *)(v23 + 80);
        }
        if ( v24 == 20 )
        {
          v25 = *(_QWORD *)(v9 + 96);
          if ( v25 )
          {
            if ( *(_QWORD *)(v25 + 32) == v23 )
              break;
          }
        }
        v22 = sub_82C230(v42);
        if ( !v22 )
          return;
      }
      if ( v19 == 24 )
      {
        v9 = *(_QWORD *)(v22 + 88);
        v19 = *(_BYTE *)(v9 + 80);
      }
      else
      {
        v9 = v22;
      }
      goto LABEL_76;
    }
  }
}
