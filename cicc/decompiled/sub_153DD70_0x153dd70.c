// Function: sub_153DD70
// Address: 0x153dd70
//
void __fastcall sub_153DD70(char *src, char *a2, __int64 a3)
{
  char *v3; // r11
  unsigned int v6; // r8d
  __int64 v7; // rcx
  unsigned int v8; // esi
  unsigned int v9; // ebx
  unsigned int v10; // r14d
  unsigned int v11; // r9d
  _BYTE *v12; // rdx
  __int64 v13; // rax
  char *v14; // rdi
  __int64 v15; // r15
  unsigned int v16; // eax
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
        v8 = 0;
        v9 = *(_DWORD *)v3;
        v10 = *((_DWORD *)v3 + 1);
        v11 = *(_DWORD *)src;
        v12 = *(_BYTE **)(v7 + 8LL * (v6 - 1));
        if ( *v12 )
        {
          v8 = 1;
          if ( (unsigned __int8)(*v12 - 4) <= 0x1Eu )
            v8 = (v12[1] != 1) + 2;
        }
        v13 = v10 - 1;
        v14 = *(char **)(v7 + 8 * v13);
        v15 = 8 * v13;
        v16 = 0;
        v17 = *v14;
        if ( *v14 )
        {
          v16 = 1;
          if ( (unsigned __int8)(v17 - 4) <= 0x1Eu )
            v16 = (v14[1] != 1) + 2;
        }
        if ( v9 < v11 || v9 == v11 && (v8 > v16 || v8 == v16 && v10 < v6) )
        {
          v24 = v3 + 8;
          if ( src != v3 )
            memmove(src + 8, src, v3 - src);
          *(_DWORD *)src = v9;
          *((_DWORD *)src + 1) = v10;
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
              if ( (unsigned __int8)(*v21 - 4) <= 0x1Eu )
                v22 = (v21[1] != 1) + 2;
            }
            v23 = 0;
            if ( v17 )
            {
              v23 = 1;
              if ( (unsigned __int8)(v17 - 4) <= 0x1Eu )
                v23 = (v14[1] != 1) + 2;
            }
            if ( v9 >= v20 && (v9 != v20 || v22 <= v23 && (v22 != v23 || v10 >= v19)) )
              break;
            v25 = *((_QWORD *)v18 - 1);
            v18 -= 8;
            *((_QWORD *)v18 + 1) = v25;
            v7 = *(_QWORD *)(a3 + 208);
            v14 = *(char **)(v7 + v15);
            v17 = *v14;
          }
          *(_DWORD *)v18 = v9;
          v24 = v3 + 8;
          *((_DWORD *)v18 + 1) = v10;
        }
        v3 = v24;
      }
      while ( a2 != v24 );
    }
  }
}
