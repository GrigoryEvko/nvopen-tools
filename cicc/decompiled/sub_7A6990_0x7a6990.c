// Function: sub_7A6990
// Address: 0x7a6990
//
__int64 __fastcall sub_7A6990(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, int a5, unsigned int a6)
{
  unsigned int v7; // r14d
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 *v10; // rax
  __int64 v11; // r13
  __int64 v12; // r11
  __int64 v14; // r14
  char v16; // al
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v21; // rbx
  unsigned __int64 i; // r15
  char v23; // al
  __int64 v24; // rax
  __int64 j; // r13
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r11
  int v29; // eax
  __int64 v30; // [rsp+8h] [rbp-58h]
  __int64 v31; // [rsp+10h] [rbp-50h]
  char v32; // [rsp+1Fh] [rbp-41h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  unsigned int v35; // [rsp+28h] [rbp-38h]
  unsigned __int64 v36; // [rsp+28h] [rbp-38h]

  v7 = a6;
  v8 = a1;
  v9 = a2;
  if ( *(_BYTE *)(a1 + 140) != 12 )
    goto LABEL_5;
  do
    v8 = *(_QWORD *)(v8 + 160);
  while ( *(_BYTE *)(v8 + 140) == 12 );
  if ( *(_BYTE *)(a2 + 140) == 12 )
  {
    do
    {
      v9 = *(_QWORD *)(v9 + 160);
LABEL_5:
      ;
    }
    while ( *(_BYTE *)(v9 + 140) == 12 );
  }
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v9 + 96LL) + 180LL) & 0x40) != 0 )
  {
    v33 = 0;
    if ( a3 )
      v33 = *(_QWORD *)(a3 + 104);
    if ( v33 == a4 )
    {
      if ( v9 == v8 )
        return 1;
      if ( dword_4F07588 )
      {
        v24 = *(_QWORD *)(v8 + 32);
        if ( *(_QWORD *)(v9 + 32) == v24 )
        {
          if ( v24 )
            return 1;
        }
      }
    }
    v10 = *(__int64 **)(v9 + 168);
    if ( a5 && (*(_BYTE *)(v9 + 176) & 0x10) != 0 )
    {
      v11 = *v10;
      v12 = 0;
    }
    else
    {
      v11 = v10[1];
      v12 = 8;
    }
    v32 = (a5 ^ 1) & 1;
    if ( v11 )
    {
      v31 = v8;
      v14 = v12;
      v30 = v9;
      do
      {
        v16 = *(_BYTE *)(v11 + 96);
        if ( (v16 & 1) != 0 || (v16 & 2) != 0 && !v32 )
        {
          v17 = a6;
          if ( dword_4D0425C )
          {
            v17 = 0;
            if ( (v16 & 2) == 0 )
              v17 = a6;
          }
          v18 = v11;
          if ( a3 )
          {
            v35 = v17;
            v19 = sub_8E5310(v11, *(_QWORD *)(a3 + 56), a3);
            v17 = v35;
            v18 = v19;
            v16 = *(_BYTE *)(v19 + 96);
          }
          if ( (v16 & 0x40) != 0
            && *(_QWORD *)(v18 + 104) <= a4
            && (unsigned int)sub_7A6990(v31, *(_QWORD *)(v18 + 40), v18, a4, 0, v17) )
          {
            return 1;
          }
        }
        v11 = *(_QWORD *)(v11 + v14);
      }
      while ( v11 );
      v7 = a6;
      v9 = v30;
      v8 = v31;
    }
    if ( v7 )
    {
      v21 = *(_QWORD *)(v9 + 160);
      for ( i = a4 - v33; v21; v21 = *(_QWORD *)(v21 + 112) )
      {
        v23 = *(_BYTE *)(v21 + 144);
        if ( (v23 & 2) == 0 )
          return 0;
        if ( (v23 & 0x40) == 0 )
        {
          for ( j = *(_QWORD *)(v21 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          if ( (unsigned int)sub_8D3410(j) )
          {
            v36 = 1;
            if ( !(unsigned int)sub_8D43F0(j) )
              v36 = sub_8D4490(j);
            for ( j = sub_8D40F0(j); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            if ( !(unsigned int)sub_8D3A70(j) || !v36 )
              continue;
          }
          else
          {
            if ( !(unsigned int)sub_8D3A70(j) )
              continue;
            v36 = 1;
          }
          v26 = *(_QWORD *)(v21 + 128);
          v27 = *(_QWORD *)(j + 128);
          if ( i >= v26 )
          {
            v28 = 0;
            while ( 1 )
            {
              if ( v26 + v27 > i )
              {
                v34 = v28;
                v29 = sub_7A6990(v8, j, 0, i - v26, 1, v7);
                v28 = v34;
                if ( v29 )
                  return 1;
              }
              if ( ++v28 < v36 )
              {
                v27 = *(_QWORD *)(j + 128);
                v26 = *(_QWORD *)(v21 + 128) + v28 * v27;
                if ( i >= v26 )
                  continue;
              }
              break;
            }
          }
        }
      }
    }
  }
  return 0;
}
