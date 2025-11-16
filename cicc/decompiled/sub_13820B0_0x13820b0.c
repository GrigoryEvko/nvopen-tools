// Function: sub_13820B0
// Address: 0x13820b0
//
void __fastcall sub_13820B0(char *src, char *a2)
{
  char *v4; // rbx
  unsigned int v5; // r13d
  unsigned int v6; // eax
  unsigned int v7; // r15d
  unsigned int v8; // r8d
  unsigned int v9; // r9d
  __int64 v10; // r10
  unsigned int v11; // edx
  unsigned int v12; // esi
  unsigned int v13; // edi
  char *v14; // rcx
  __int64 v15; // [rsp-48h] [rbp-48h]
  unsigned int v16; // [rsp-40h] [rbp-40h]
  unsigned int v17; // [rsp-3Ch] [rbp-3Ch]

  if ( src != a2 )
  {
    v4 = src + 24;
    if ( a2 != src + 24 )
    {
      do
      {
        v5 = *(_DWORD *)v4;
        v6 = *(_DWORD *)src;
        v7 = *((_DWORD *)v4 + 1);
        v8 = *((_DWORD *)v4 + 2);
        v9 = *((_DWORD *)v4 + 3);
        v10 = *((_QWORD *)v4 + 2);
        if ( *(_DWORD *)v4 < *(_DWORD *)src
          || (v11 = *((_DWORD *)src + 1), v7 < v11) && v5 == v6
          || (v12 = *((_DWORD *)src + 2), v13 = *((_DWORD *)src + 3), v5 <= v6)
          && (v7 <= v11 || v5 != v6)
          && (v12 > v8 || v13 > v9 && v12 == v8 || v12 >= v8 && (v13 >= v9 || v12 != v8) && *((_QWORD *)src + 2) > v10) )
        {
          v14 = v4 + 24;
          if ( src != v4 )
          {
            v16 = *((_DWORD *)v4 + 3);
            v15 = *((_QWORD *)v4 + 2);
            v17 = *((_DWORD *)v4 + 2);
            memmove(src + 24, src, v4 - src);
            v14 = v4 + 24;
            v9 = v16;
            v10 = v15;
            v8 = v17;
          }
          *(_DWORD *)src = v5;
          *((_DWORD *)src + 1) = v7;
          *((_DWORD *)src + 2) = v8;
          *((_DWORD *)src + 3) = v9;
          *((_QWORD *)src + 2) = v10;
        }
        else
        {
          sub_1382000((unsigned int *)v4);
          v14 = v4 + 24;
        }
        v4 = v14;
      }
      while ( a2 != v14 );
    }
  }
}
