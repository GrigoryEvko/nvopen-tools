// Function: sub_16C1870
// Address: 0x16c1870
//
unsigned __int64 __fastcall sub_16C1870(int *a1, int *a2, size_t a3)
{
  unsigned int v5; // edi
  unsigned __int64 v6; // rbx
  int v7; // eax
  unsigned int v8; // edx
  unsigned __int64 result; // rax
  unsigned int v10; // edi
  int *v11; // r13
  int *v12; // r14
  unsigned __int64 v13; // rsi
  signed __int64 v14; // r13
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // r13
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r9
  char *v21; // rdi
  char *v22; // r11
  unsigned int v23; // edi
  unsigned int v24; // edi
  unsigned int v25; // eax
  __int64 v26; // rcx

  v5 = a1[5];
  v6 = a3;
  v7 = a1[4];
  v8 = (v5 + a3) & 0x1FFFFFFF;
  a1[5] = v8;
  result = (unsigned int)(a3 >> 29) + (v8 < v5) + v7;
  a1[4] = result;
  v10 = v5 & 0x3F;
  if ( !v10 )
  {
    v11 = a2;
    v12 = a1 + 6;
    if ( v6 <= 0x3F )
      goto LABEL_3;
    goto LABEL_19;
  }
  v16 = v10;
  v17 = 64 - v16;
  v18 = (_QWORD *)((char *)a1 + v10 + 24);
  if ( a3 < 64 - v16 )
    return (unsigned __int64)memcpy(v18, a2, a3);
  if ( (unsigned int)v17 >= 8 )
  {
    v20 = (unsigned __int64)(v18 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v18 = *(_QWORD *)a2;
    *(_QWORD *)((char *)v18 + (unsigned int)v17 - 8) = *(_QWORD *)((char *)a2 + (unsigned int)v17 - 8);
    v21 = (char *)v18 - v20;
    v22 = (char *)((char *)a2 - v21);
    v23 = (v17 + (_DWORD)v21) & 0xFFFFFFF8;
    if ( v23 >= 8 )
    {
      v24 = v23 & 0xFFFFFFF8;
      v25 = 0;
      do
      {
        v26 = v25;
        v25 += 8;
        *(_QWORD *)(v20 + v26) = *(_QWORD *)&v22[v26];
      }
      while ( v25 < v24 );
    }
  }
  else if ( (v17 & 4) != 0 )
  {
    *(_DWORD *)v18 = *a2;
    *(_DWORD *)((char *)v18 + (unsigned int)v17 - 4) = *(int *)((char *)a2 + (unsigned int)v17 - 4);
  }
  else if ( (_DWORD)v17 )
  {
    *(_BYTE *)v18 = *(_BYTE *)a2;
    if ( (v17 & 2) != 0 )
      *(_WORD *)((char *)v18 + (unsigned int)v17 - 2) = *(_WORD *)((char *)a2 + (unsigned int)v17 - 2);
  }
  v12 = a1 + 6;
  v11 = (int *)((char *)a2 + v17);
  v6 = v16 + a3 - 64;
  result = (unsigned __int64)sub_16C10A0(a1, a1 + 6, 64);
  if ( v6 > 0x3F )
  {
LABEL_19:
    v19 = v6;
    LODWORD(v6) = v6 & 0x3F;
    result = (unsigned __int64)sub_16C10A0(a1, v11, v19 & 0xFFFFFFFFFFFFFFC0LL);
    v11 = (int *)result;
  }
LABEL_3:
  if ( (unsigned int)v6 < 8 )
  {
    if ( (v6 & 4) != 0 )
    {
      *v12 = *v11;
      result = *(unsigned int *)((char *)v11 + (unsigned int)v6 - 4);
      *(int *)((char *)v12 + (unsigned int)v6 - 4) = result;
    }
    else if ( (_DWORD)v6 )
    {
      result = *(unsigned __int8 *)v11;
      *(_BYTE *)v12 = result;
      if ( (v6 & 2) != 0 )
      {
        result = *(unsigned __int16 *)((char *)v11 + (unsigned int)v6 - 2);
        *(_WORD *)((char *)v12 + (unsigned int)v6 - 2) = result;
      }
    }
  }
  else
  {
    v13 = (unsigned __int64)(v12 + 2) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v12 = *(_QWORD *)v11;
    result = (unsigned int)v6;
    *(_QWORD *)((char *)v12 + (unsigned int)v6 - 8) = *(_QWORD *)((char *)v11 + (unsigned int)v6 - 8);
    v14 = (char *)v11 - ((char *)v12 - v13);
    if ( (((_DWORD)v6 + (_DWORD)v12 - (_DWORD)v13) & 0xFFFFFFF8) >= 8 )
    {
      LODWORD(result) = 0;
      do
      {
        v15 = (unsigned int)result;
        result = (unsigned int)(result + 8);
        *(_QWORD *)(v13 + v15) = *(_QWORD *)(v14 + v15);
      }
      while ( (unsigned int)result < (((_DWORD)v6 + (_DWORD)v12 - (_DWORD)v13) & 0xFFFFFFF8) );
    }
  }
  return result;
}
