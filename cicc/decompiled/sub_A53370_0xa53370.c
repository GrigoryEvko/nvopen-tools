// Function: sub_A53370
// Address: 0xa53370
//
__int16 __fastcall sub_A53370(__int64 a1, const void *a2, size_t a3, char a4, unsigned __int16 a5)
{
  size_t v5; // rax
  __int64 v8; // r12
  _WORD *v9; // rdi
  unsigned __int64 v10; // rax
  const char *v11; // r13
  size_t v12; // rax
  _QWORD *v13; // rcx
  size_t v14; // rdx
  unsigned __int64 v15; // rdi
  char *v16; // rcx
  const char *v17; // r13
  unsigned int v18; // ecx
  unsigned int v19; // ecx
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax

  LOWORD(v5) = HIBYTE(a5);
  if ( !HIBYTE(a5) || a4 != (_BYTE)a5 )
  {
    v8 = *(_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 8) )
      *(_BYTE *)(a1 + 8) = 0;
    else
      v8 = sub_904010(*(_QWORD *)a1, *(const char **)(a1 + 16));
    v9 = *(_WORD **)(v8 + 32);
    v10 = *(_QWORD *)(v8 + 24) - (_QWORD)v9;
    if ( v10 < a3 )
    {
      v21 = sub_CB6200(v8, a2, a3);
      v9 = *(_WORD **)(v21 + 32);
      v8 = v21;
      v10 = *(_QWORD *)(v21 + 24) - (_QWORD)v9;
    }
    else if ( a3 )
    {
      memcpy(v9, a2, a3);
      v22 = *(_QWORD *)(v8 + 24);
      v9 = (_WORD *)(a3 + *(_QWORD *)(v8 + 32));
      *(_QWORD *)(v8 + 32) = v9;
      v10 = v22 - (_QWORD)v9;
    }
    if ( v10 <= 1 )
    {
      v8 = sub_CB6200(v8, ": ", 2);
    }
    else
    {
      *v9 = 8250;
      *(_QWORD *)(v8 + 32) += 2LL;
    }
    v11 = "true";
    if ( !a4 )
      v11 = "false";
    v12 = strlen(v11);
    v13 = *(_QWORD **)(v8 + 32);
    v14 = v12;
    v5 = *(_QWORD *)(v8 + 24) - (_QWORD)v13;
    if ( v14 <= v5 )
    {
      if ( (unsigned int)v14 >= 8 )
      {
        v15 = (unsigned __int64)(v13 + 1) & 0xFFFFFFFFFFFFFFF8LL;
        *v13 = *(_QWORD *)v11;
        LOWORD(v5) = v14;
        *(_QWORD *)((char *)v13 + (unsigned int)v14 - 8) = *(_QWORD *)&v11[(unsigned int)v14 - 8];
        v16 = (char *)v13 - v15;
        v17 = (const char *)(v11 - v16);
        v18 = (v14 + (_DWORD)v16) & 0xFFFFFFF8;
        if ( v18 >= 8 )
        {
          v19 = v18 & 0xFFFFFFF8;
          LODWORD(v5) = 0;
          do
          {
            v20 = (unsigned int)v5;
            LODWORD(v5) = v5 + 8;
            *(_QWORD *)(v15 + v20) = *(_QWORD *)&v17[v20];
          }
          while ( (unsigned int)v5 < v19 );
        }
      }
      else if ( (v14 & 4) != 0 )
      {
        *(_DWORD *)v13 = *(_DWORD *)v11;
        LOWORD(v5) = v14;
        *(_DWORD *)((char *)v13 + (unsigned int)v14 - 4) = *(_DWORD *)&v11[(unsigned int)v14 - 4];
      }
      else if ( (_DWORD)v14 )
      {
        LOWORD(v5) = *(unsigned __int8 *)v11;
        *(_BYTE *)v13 = v5;
        if ( (v14 & 2) != 0 )
        {
          LOWORD(v5) = v14;
          *(_WORD *)((char *)v13 + (unsigned int)v14 - 2) = *(_WORD *)&v11[(unsigned int)v14 - 2];
        }
      }
      *(_QWORD *)(v8 + 32) += v14;
    }
    else
    {
      LOWORD(v5) = sub_CB6200(v8, v11, v14);
    }
  }
  return v5;
}
