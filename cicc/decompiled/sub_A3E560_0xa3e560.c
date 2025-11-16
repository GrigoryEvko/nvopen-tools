// Function: sub_A3E560
// Address: 0xa3e560
//
void __fastcall sub_A3E560(char *src, char *a2, __int64 a3)
{
  char *v3; // r11
  unsigned int v6; // r8d
  __int64 v7; // rcx
  unsigned int v8; // ebx
  unsigned int v9; // r14d
  unsigned int v10; // r9d
  _BYTE *v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // rdx
  unsigned int v14; // esi
  char *v15; // rdi
  __int64 v16; // r15
  char v17; // dl
  char *v18; // rax
  unsigned int v19; // r9d
  unsigned int v20; // r10d
  _BYTE *v21; // r8
  unsigned int v22; // esi
  unsigned int v23; // ecx
  char *v24; // r15
  __int64 v25; // rdx

  if ( src != a2 )
  {
    v3 = src + 8;
    if ( a2 != src + 8 )
    {
      do
      {
        v6 = *((_DWORD *)src + 1);
        v7 = *(_QWORD *)(a3 + 208);
        v8 = *(_DWORD *)v3;
        v9 = *((_DWORD *)v3 + 1);
        v10 = *(_DWORD *)src;
        v11 = *(_BYTE **)(v7 + 8LL * (v6 - 1));
        v12 = 0;
        if ( *v11 )
        {
          v12 = 1;
          if ( (unsigned __int8)(*v11 - 5) <= 0x1Fu )
            v12 = ((v11[1] & 0x7F) != 1) + 2;
        }
        v13 = v9 - 1;
        v14 = 0;
        v15 = *(char **)(v7 + 8 * v13);
        v16 = 8 * v13;
        v17 = *v15;
        if ( *v15 )
        {
          v14 = 1;
          if ( (unsigned __int8)(v17 - 5) <= 0x1Fu )
            v14 = ((v15[1] & 0x7F) != 1) + 2;
        }
        if ( v8 < v10 || v8 == v10 && (v14 < v12 || v14 == v12 && v9 < v6) )
        {
          v24 = v3 + 8;
          if ( src != v3 )
            memmove(src + 8, src, v3 - src);
          *(_DWORD *)src = v8;
          *((_DWORD *)src + 1) = v9;
        }
        else
        {
          v18 = v3;
          while ( 1 )
          {
            v19 = *((_DWORD *)v18 - 1);
            v20 = *((_DWORD *)v18 - 2);
            v21 = *(_BYTE **)(v7 + 8LL * (v19 - 1));
            v22 = 0;
            if ( *v21 )
            {
              v22 = 1;
              if ( (unsigned __int8)(*v21 - 5) <= 0x1Fu )
                v22 = ((v21[1] & 0x7F) != 1) + 2;
            }
            v23 = 0;
            if ( v17 )
            {
              v23 = 1;
              if ( (unsigned __int8)(v17 - 5) <= 0x1Fu )
                v23 = ((v15[1] & 0x7F) != 1) + 2;
            }
            if ( v8 >= v20 && (v8 != v20 || v23 >= v22 && (v23 != v22 || v9 >= v19)) )
              break;
            v25 = *((_QWORD *)v18 - 1);
            v18 -= 8;
            *((_QWORD *)v18 + 1) = v25;
            v7 = *(_QWORD *)(a3 + 208);
            v15 = *(char **)(v7 + v16);
            v17 = *v15;
          }
          *(_DWORD *)v18 = v8;
          v24 = v3 + 8;
          *((_DWORD *)v18 + 1) = v9;
        }
        v3 = v24;
      }
      while ( a2 != v24 );
    }
  }
}
