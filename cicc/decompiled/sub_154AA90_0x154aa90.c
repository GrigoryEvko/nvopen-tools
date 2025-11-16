// Function: sub_154AA90
// Address: 0x154aa90
//
void __fastcall sub_154AA90(__int64 a1, const char *a2, size_t a3, char a4, _BYTE *a5)
{
  __int64 v7; // r12
  _WORD *v8; // rdi
  unsigned __int64 v9; // rax
  const char *v10; // r13
  size_t v11; // rax
  _QWORD *v12; // rcx
  size_t v13; // rdx
  unsigned __int64 v14; // rdi
  char *v15; // rcx
  const char *v16; // r13
  unsigned int v17; // ecx
  unsigned int v18; // ecx
  unsigned int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax

  if ( !a5[1] || *a5 != a4 )
  {
    v7 = *(_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 8) )
      *(_BYTE *)(a1 + 8) = 0;
    else
      v7 = sub_1263B40(*(_QWORD *)a1, *(const char **)(a1 + 16));
    v8 = *(_WORD **)(v7 + 24);
    v9 = *(_QWORD *)(v7 + 16) - (_QWORD)v8;
    if ( v9 < a3 )
    {
      v21 = sub_16E7EE0(v7, a2, a3);
      v8 = *(_WORD **)(v21 + 24);
      v7 = v21;
      v9 = *(_QWORD *)(v21 + 16) - (_QWORD)v8;
    }
    else if ( a3 )
    {
      memcpy(v8, a2, a3);
      v22 = *(_QWORD *)(v7 + 16);
      v8 = (_WORD *)(a3 + *(_QWORD *)(v7 + 24));
      *(_QWORD *)(v7 + 24) = v8;
      v9 = v22 - (_QWORD)v8;
    }
    if ( v9 <= 1 )
    {
      v7 = sub_16E7EE0(v7, ": ", 2);
    }
    else
    {
      *v8 = 8250;
      *(_QWORD *)(v7 + 24) += 2LL;
    }
    v10 = "true";
    if ( !a4 )
      v10 = "false";
    v11 = strlen(v10);
    v12 = *(_QWORD **)(v7 + 24);
    v13 = v11;
    if ( v11 <= *(_QWORD *)(v7 + 16) - (_QWORD)v12 )
    {
      if ( (unsigned int)v11 >= 8 )
      {
        v14 = (unsigned __int64)(v12 + 1) & 0xFFFFFFFFFFFFFFF8LL;
        *v12 = *(_QWORD *)v10;
        *(_QWORD *)((char *)v12 + (unsigned int)v11 - 8) = *(_QWORD *)&v10[(unsigned int)v11 - 8];
        v15 = (char *)v12 - v14;
        v16 = (const char *)(v10 - v15);
        v17 = (v11 + (_DWORD)v15) & 0xFFFFFFF8;
        if ( v17 >= 8 )
        {
          v18 = v17 & 0xFFFFFFF8;
          v19 = 0;
          do
          {
            v20 = v19;
            v19 += 8;
            *(_QWORD *)(v14 + v20) = *(_QWORD *)&v16[v20];
          }
          while ( v19 < v18 );
        }
      }
      else if ( (v11 & 4) != 0 )
      {
        *(_DWORD *)v12 = *(_DWORD *)v10;
        *(_DWORD *)((char *)v12 + (unsigned int)v11 - 4) = *(_DWORD *)&v10[(unsigned int)v11 - 4];
      }
      else if ( (_DWORD)v11 )
      {
        *(_BYTE *)v12 = *v10;
        if ( (v11 & 2) != 0 )
          *(_WORD *)((char *)v12 + (unsigned int)v11 - 2) = *(_WORD *)&v10[(unsigned int)v11 - 2];
      }
      *(_QWORD *)(v7 + 24) += v13;
    }
    else
    {
      sub_16E7EE0(v7, v10);
    }
  }
}
