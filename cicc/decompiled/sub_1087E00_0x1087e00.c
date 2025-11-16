// Function: sub_1087E00
// Address: 0x1087e00
//
_QWORD *__fastcall sub_1087E00(_BYTE *a1)
{
  _QWORD *v1; // r13
  _QWORD *result; // rax
  unsigned __int64 v4; // r12
  unsigned int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // r9
  unsigned __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  _DWORD *v11; // rdx
  _DWORD *v12; // r15
  unsigned int v13; // r10d
  _DWORD *v14; // rcx
  __int64 v15; // r9
  char *v16; // rdx
  unsigned __int64 v17; // rdi
  char *v18; // rax
  __int64 v19; // rsi
  char *v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // r8
  _DWORD *v23; // rdi
  char *v24; // r9
  _DWORD *v25; // rcx
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rdx
  signed __int64 v28; // r9
  unsigned int v29; // esi
  __int64 v30; // r8
  __int64 v31; // [rsp+0h] [rbp-50h]
  int v32; // [rsp+8h] [rbp-48h]
  _QWORD *i; // [rsp+10h] [rbp-40h]
  unsigned __int64 v34; // [rsp+18h] [rbp-38h]
  __int64 v35; // [rsp+18h] [rbp-38h]

  v1 = *(_QWORD **)(*(_QWORD *)a1 + 8LL);
  result = &v1[5 * *(unsigned int *)(*(_QWORD *)a1 + 16LL)];
  for ( i = result; i != v1; v1 += 5 )
  {
    v4 = (-(__int64)(a1[240] == 0) & 0xFFFFFFFFFFFFFFFELL) + 20;
    v5 = a1[240] == 0 ? 18 : 20;
    v34 = (v4 + v1[1] - 1) / v4;
    v6 = sub_1084C60(a1, (__int64)".file", 5u);
    v7 = v34;
    v8 = *(unsigned int *)(v6 + 72);
    *(_DWORD *)(v6 + 12) = -2;
    v9 = v6;
    *(_BYTE *)(v6 + 18) = 103;
    if ( (unsigned int)v34 == v8 )
    {
      v11 = *(_DWORD **)(v6 + 64);
      v12 = &v11[6 * (unsigned int)v34];
    }
    else
    {
      v10 = 24LL * (unsigned int)v34;
      if ( (unsigned int)v34 >= v8 )
      {
        if ( (unsigned int)v34 > (unsigned __int64)*(unsigned int *)(v9 + 76) )
        {
          v31 = 24LL * (unsigned int)v34;
          v32 = v34;
          v35 = v9;
          sub_C8D5F0(v9 + 64, (const void *)(v9 + 80), (unsigned int)v7, 0x18u, v9, v7);
          v9 = v35;
          v10 = v31;
          LODWORD(v7) = v32;
          v8 = *(unsigned int *)(v35 + 72);
        }
        v11 = *(_DWORD **)(v9 + 64);
        v25 = &v11[6 * v8];
        v12 = (_DWORD *)((char *)v11 + v10);
        if ( v25 != (_DWORD *)((char *)v11 + v10) )
        {
          do
          {
            if ( v25 )
            {
              *((_QWORD *)v25 + 2) = 0;
              *(_OWORD *)v25 = 0;
            }
            v25 += 6;
          }
          while ( v12 != v25 );
          v11 = *(_DWORD **)(v9 + 64);
          v12 = (_DWORD *)((char *)v11 + v10);
        }
        *(_DWORD *)(v9 + 72) = v7;
      }
      else
      {
        v11 = *(_DWORD **)(v9 + 64);
        *(_DWORD *)(v9 + 72) = v34;
        v12 = (_DWORD *)((char *)v11 + v10);
      }
    }
    result = (_QWORD *)v1[1];
    v13 = (unsigned int)result;
    if ( v11 != v12 )
    {
      *v11 = 1;
      v14 = v11 + 6;
      LODWORD(v15) = 0;
      if ( v5 >= (unsigned int)result )
      {
        v15 = 0;
LABEL_11:
        v23 = v11 + 1;
        v24 = (char *)(*v1 + v15);
        if ( v13 >= 8 )
        {
          v26 = (unsigned __int64)(v11 + 3);
          *(_QWORD *)(v26 - 8) = *(_QWORD *)v24;
          v27 = v26 & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)((char *)v23 + v13 - 8) = *(_QWORD *)&v24[v13 - 8];
          v28 = v24 - ((char *)v23 - v27);
          if ( ((v13 + (_DWORD)v23 - (_DWORD)v27) & 0xFFFFFFF8) >= 8 )
          {
            v29 = 0;
            do
            {
              v30 = v29;
              v29 += 8;
              *(_QWORD *)(v27 + v30) = *(_QWORD *)(v28 + v30);
            }
            while ( v29 < ((v13 + (_DWORD)v23 - (_DWORD)v27) & 0xFFFFFFF8) );
          }
        }
        else if ( (v13 & 4) != 0 )
        {
          *v23 = *(_DWORD *)v24;
          *(_DWORD *)((char *)v11 + v13) = *(_DWORD *)&v24[v13 - 4];
        }
        else if ( v13 )
        {
          *(_BYTE *)v23 = *v24;
          if ( (v13 & 2) != 0 )
            *(_WORD *)((char *)v23 + v13 - 2) = *(_WORD *)&v24[v13 - 2];
        }
        result = memset((char *)v23 + v13, 0, v5 - v13);
      }
      else
      {
        while ( 1 )
        {
          v16 = (char *)(*v1 + (unsigned int)v15);
          v17 = (unsigned __int64)(v14 - 3) & 0xFFFFFFFFFFFFFFF8LL;
          v18 = (char *)v14 - v17 - 20;
          *(_QWORD *)(v14 - 5) = *(_QWORD *)v16;
          v19 = *(_QWORD *)&v16[(unsigned int)v4 - 8];
          v20 = (char *)(v16 - v18);
          result = (_QWORD *)(((_DWORD)v4 + (_DWORD)v18) & 0xFFFFFFF8);
          *(_QWORD *)((char *)v14 + (unsigned int)v4 - 28) = v19;
          v21 = 0;
          do
          {
            v22 = v21;
            v21 += 8;
            *(_QWORD *)(v17 + v22) = *(_QWORD *)&v20[v22];
          }
          while ( v21 < (unsigned int)result );
          v13 -= v5;
          v15 = v5 + (unsigned int)v15;
          v11 = v14;
          if ( v14 == v12 )
            break;
          *v14 = 1;
          v14 += 6;
          if ( v5 >= v13 )
            goto LABEL_11;
        }
      }
    }
  }
  return result;
}
