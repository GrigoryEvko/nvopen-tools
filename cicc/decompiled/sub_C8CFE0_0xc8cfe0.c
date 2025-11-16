// Function: sub_C8CFE0
// Address: 0xc8cfe0
//
void __fastcall sub_C8CFE0(__int64 a1, void *a2, void *a3, __int64 a4)
{
  char v5; // al
  __int64 *v8; // rsi
  __int64 v9; // rdx
  unsigned int v10; // r9d
  __int64 *v11; // rax
  __int64 v12; // r8
  __int64 *v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 *v16; // r10
  __int64 *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rsi
  int v20; // eax
  const void *v22; // rsi
  size_t v23; // rdx
  int v24; // eax
  int v25; // eax
  int v26; // edx
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // edx
  int v31; // edx
  int v32; // edx
  size_t v33; // rdx
  int v34; // eax
  int v35; // edx
  int v36; // eax
  int v37; // edx
  int v38; // eax
  __int64 v39; // rax

  if ( a1 != a4 )
  {
    v5 = *(_BYTE *)(a4 + 28);
    if ( *(_BYTE *)(a1 + 28) )
    {
      v8 = *(__int64 **)(a1 + 8);
      v9 = *(unsigned int *)(a1 + 20);
      if ( v5 )
      {
        v10 = *(_DWORD *)(a1 + 20);
        if ( *(_DWORD *)(a4 + 20) <= (unsigned int)v9 )
          v10 = *(_DWORD *)(a4 + 20);
        v11 = *(__int64 **)(a4 + 8);
        v12 = v10;
        v13 = &v11[v12];
        if ( v12 * 8 )
        {
          do
          {
            v14 = *v8;
            v15 = *v11++;
            *v8++ = v15;
            *(v11 - 1) = v14;
          }
          while ( v11 != v13 );
          v8 = *(__int64 **)(a1 + 8);
          LODWORD(v9) = *(_DWORD *)(a1 + 20);
          v11 = *(__int64 **)(a4 + 8);
          v16 = &v8[v12];
        }
        else
        {
          v16 = *(__int64 **)(a1 + 8);
        }
        v17 = &v11[v12];
        if ( (unsigned int)v9 <= v10 )
        {
          v19 = *(unsigned int *)(a4 + 20);
          if ( v17 == &v11[v19] )
            goto LABEL_15;
          memmove(v16, v17, 8 * v19 - v12 * 8);
          LODWORD(v9) = *(_DWORD *)(a1 + 20);
        }
        else
        {
          v18 = (unsigned int)v9;
          if ( v16 != &v8[v18] )
          {
            memmove(v17, v16, v18 * 8 - v12 * 8);
            LODWORD(v9) = *(_DWORD *)(a1 + 20);
            LODWORD(v19) = *(_DWORD *)(a4 + 20);
LABEL_15:
            *(_DWORD *)(a1 + 20) = v19;
            *(_DWORD *)(a4 + 20) = v9;
            v20 = *(_DWORD *)(a1 + 24);
            *(_DWORD *)(a1 + 24) = *(_DWORD *)(a4 + 24);
            *(_DWORD *)(a4 + 24) = v20;
            return;
          }
        }
        LODWORD(v19) = *(_DWORD *)(a4 + 20);
        goto LABEL_15;
      }
      v33 = 8 * v9;
      if ( v33 )
        a3 = memmove(a3, v8, v33);
      v34 = *(_DWORD *)(a4 + 16);
      *(_DWORD *)(a4 + 16) = *(_DWORD *)(a1 + 16);
      v35 = *(_DWORD *)(a1 + 20);
      *(_DWORD *)(a1 + 16) = v34;
      v36 = *(_DWORD *)(a4 + 20);
      *(_DWORD *)(a4 + 20) = v35;
      v37 = *(_DWORD *)(a1 + 24);
      *(_DWORD *)(a1 + 20) = v36;
      v38 = *(_DWORD *)(a4 + 24);
      *(_DWORD *)(a4 + 24) = v37;
      *(_DWORD *)(a1 + 24) = v38;
      v39 = *(_QWORD *)(a4 + 8);
      *(_BYTE *)(a1 + 28) = 0;
      *(_QWORD *)(a1 + 8) = v39;
      *(_QWORD *)(a4 + 8) = a3;
      *(_BYTE *)(a4 + 28) = 1;
    }
    else
    {
      v22 = *(const void **)(a4 + 8);
      if ( v5 )
      {
        v23 = 8LL * *(unsigned int *)(a4 + 20);
        if ( v23 )
          a2 = memmove(a2, v22, v23);
        v24 = *(_DWORD *)(a4 + 16);
        *(_DWORD *)(a4 + 16) = *(_DWORD *)(a1 + 16);
        *(_DWORD *)(a1 + 16) = v24;
        v25 = *(_DWORD *)(a1 + 20);
        *(_DWORD *)(a1 + 20) = *(_DWORD *)(a4 + 20);
        v26 = *(_DWORD *)(a4 + 24);
        *(_DWORD *)(a4 + 20) = v25;
        v27 = *(_DWORD *)(a1 + 24);
        *(_DWORD *)(a1 + 24) = v26;
        *(_DWORD *)(a4 + 24) = v27;
        v28 = *(_QWORD *)(a1 + 8);
        *(_BYTE *)(a4 + 28) = 0;
        *(_QWORD *)(a4 + 8) = v28;
        *(_QWORD *)(a1 + 8) = a2;
        *(_BYTE *)(a1 + 28) = 1;
      }
      else
      {
        v29 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 8) = v22;
        v30 = *(_DWORD *)(a4 + 16);
        *(_QWORD *)(a4 + 8) = v29;
        LODWORD(v29) = *(_DWORD *)(a1 + 16);
        *(_DWORD *)(a1 + 16) = v30;
        v31 = *(_DWORD *)(a4 + 20);
        *(_DWORD *)(a4 + 16) = v29;
        LODWORD(v29) = *(_DWORD *)(a1 + 20);
        *(_DWORD *)(a1 + 20) = v31;
        v32 = *(_DWORD *)(a4 + 24);
        *(_DWORD *)(a4 + 20) = v29;
        LODWORD(v29) = *(_DWORD *)(a1 + 24);
        *(_DWORD *)(a1 + 24) = v32;
        *(_DWORD *)(a4 + 24) = v29;
      }
    }
  }
}
