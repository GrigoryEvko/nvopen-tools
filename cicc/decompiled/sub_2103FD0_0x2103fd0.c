// Function: sub_2103FD0
// Address: 0x2103fd0
//
__int64 __fastcall sub_2103FD0(__int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r8
  unsigned __int64 v4; // r12
  unsigned __int64 i; // rbx
  __int64 v6; // r15
  __int64 v7; // r9
  __int64 v8; // r13
  int v9; // eax
  __int64 v10; // rsi
  _WORD *v11; // rsi
  _WORD *v12; // rdi
  unsigned __int64 v13; // rcx
  _WORD *v14; // rsi
  _QWORD *v15; // rax
  int v16; // eax
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 v20; // r9
  __int64 result; // rax
  char v22; // dl
  int v23; // esi
  __int64 v24; // rcx
  __int64 v25; // r10
  _WORD *v26; // r11
  int v27; // ecx
  _WORD *v28; // r10
  unsigned __int64 v29; // rcx
  _WORD *v30; // rsi
  _QWORD *v31; // rdx
  int v32; // edx
  __int64 v33; // r9
  __int64 v34; // rdx
  unsigned __int64 v35; // [rsp+0h] [rbp-40h]
  __int64 v36; // [rsp+8h] [rbp-38h]

  v2 = a2;
  v4 = a2;
  for ( i = a2; (*(_BYTE *)(v4 + 46) & 4) != 0; v4 = *(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL )
    ;
  v6 = *(_QWORD *)(a2 + 24) + 24LL;
  do
  {
    v7 = *(_QWORD *)(v4 + 32);
    v8 = v7 + 40LL * *(unsigned int *)(v4 + 40);
    if ( v7 != v8 )
      break;
    v4 = *(_QWORD *)(v4 + 8);
    if ( v6 == v4 )
      break;
  }
  while ( (*(_BYTE *)(v4 + 46) & 4) != 0 );
  if ( v7 != v8 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)v7 )
      {
        if ( *(_BYTE *)v7 == 12 )
        {
          v35 = v2;
          v36 = v7;
          sub_2103E80(a1, *(_QWORD *)(v7 + 24));
          v2 = v35;
          v18 = v8;
          v17 = v36 + 40;
          if ( v36 + 40 == v8 )
            goto LABEL_21;
          goto LABEL_51;
        }
      }
      else if ( (*(_BYTE *)(v7 + 3) & 0x10) != 0 && (*(_BYTE *)(v7 + 4) & 8) == 0 )
      {
        v9 = *(_DWORD *)(v7 + 8);
        if ( v9 > 0 )
        {
          v10 = *a1;
          if ( !*a1 )
            BUG();
          v13 = v9 * (*(_DWORD *)(*(_QWORD *)(v10 + 8) + 24LL * (unsigned int)v9 + 16) & 0xFu);
          v11 = (_WORD *)(*(_QWORD *)(v10 + 56)
                        + 2LL * (*(_DWORD *)(*(_QWORD *)(v10 + 8) + 24LL * (unsigned int)v9 + 16) >> 4));
          v12 = v11 + 1;
          LOWORD(v13) = *v11 + v13;
          while ( 1 )
          {
            v14 = v12;
            if ( !v12 )
              break;
            while ( 1 )
            {
              ++v14;
              v15 = (_QWORD *)(a1[1] + ((v13 >> 3) & 0x1FF8));
              *v15 &= ~(1LL << v13);
              v16 = (unsigned __int16)*(v14 - 1);
              v12 = 0;
              if ( !(_WORD)v16 )
                break;
              v13 = (unsigned int)(v16 + v13);
              if ( !v14 )
                goto LABEL_17;
            }
          }
        }
      }
LABEL_17:
      v17 = v7 + 40;
      v18 = v8;
      if ( v17 == v8 )
      {
LABEL_21:
        while ( 1 )
        {
          v4 = *(_QWORD *)(v4 + 8);
          if ( v6 == v4 || (*(_BYTE *)(v4 + 46) & 4) == 0 )
            break;
          v8 = *(_QWORD *)(v4 + 32);
          v18 = v8 + 40LL * *(unsigned int *)(v4 + 40);
          if ( v8 != v18 )
            goto LABEL_52;
        }
        v7 = v8;
        v8 = v18;
        if ( v7 == v18 )
          break;
      }
      else
      {
LABEL_51:
        v8 = v17;
LABEL_52:
        v7 = v8;
        v8 = v18;
      }
    }
  }
  if ( (*(_BYTE *)(v2 + 46) & 4) != 0 )
  {
    do
      i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(i + 46) & 4) != 0 );
  }
  v19 = *(_QWORD *)(v2 + 24) + 24LL;
  do
  {
    v20 = *(_QWORD *)(i + 32);
    result = v20 + 40LL * *(unsigned int *)(i + 40);
    if ( v20 != result )
      break;
    i = *(_QWORD *)(i + 8);
    if ( v19 == i )
      break;
  }
  while ( (*(_BYTE *)(i + 46) & 4) != 0 );
  if ( result != v20 )
  {
    do
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v20 )
        {
          v22 = *(_BYTE *)(v20 + 4);
          if ( (v22 & 1) == 0
            && (v22 & 2) == 0
            && ((*(_BYTE *)(v20 + 3) & 0x10) == 0 || (*(_DWORD *)v20 & 0xFFF00) != 0)
            && (*(_BYTE *)(v20 + 4) & 8) == 0 )
          {
            v23 = *(_DWORD *)(v20 + 8);
            if ( v23 > 0 )
            {
              v24 = *a1;
              if ( !*a1 )
                BUG();
              v25 = *(_QWORD *)(v24 + 8);
              v26 = (_WORD *)(*(_QWORD *)(v24 + 56) + 2LL * (*(_DWORD *)(v25 + 24LL * (unsigned int)v23 + 16) >> 4));
              v27 = *(_DWORD *)(v25 + 24LL * (unsigned int)v23 + 16) & 0xF;
              v28 = v26 + 1;
              v29 = (unsigned int)(v23 * v27);
              LOWORD(v29) = *v26 + v29;
              while ( 1 )
              {
                v30 = v28;
                if ( !v28 )
                  break;
                while ( 1 )
                {
                  ++v30;
                  v31 = (_QWORD *)(a1[1] + ((v29 >> 3) & 0x1FF8));
                  *v31 |= 1LL << v29;
                  v32 = (unsigned __int16)*(v30 - 1);
                  v28 = 0;
                  if ( !(_WORD)v32 )
                    break;
                  v29 = (unsigned int)(v32 + v29);
                  if ( !v30 )
                    goto LABEL_42;
                }
              }
            }
          }
        }
LABEL_42:
        v33 = v20 + 40;
        v34 = result;
        if ( v33 == result )
          break;
        result = v33;
LABEL_53:
        v20 = result;
        result = v34;
      }
      while ( 1 )
      {
        i = *(_QWORD *)(i + 8);
        if ( v19 == i || (*(_BYTE *)(i + 46) & 4) == 0 )
          break;
        result = *(_QWORD *)(i + 32);
        v34 = result + 40LL * *(unsigned int *)(i + 40);
        if ( result != v34 )
          goto LABEL_53;
      }
      v20 = result;
      result = v34;
    }
    while ( v34 != v20 );
  }
  return result;
}
