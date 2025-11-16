// Function: sub_E13D60
// Address: 0xe13d60
//
__int64 __fastcall sub_E13D60(__int64 a1, char **a2)
{
  char *v4; // rbx
  int v5; // eax
  _BYTE *v6; // r13
  int v7; // r13d
  int i; // ebx
  char *v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  _BYTE *v15; // r15
  unsigned int v17; // [rsp+8h] [rbp-38h]
  int v18; // [rsp+Ch] [rbp-34h]

  v4 = a2[1];
  v17 = *((_DWORD *)a2 + 6);
  v5 = *((_DWORD *)a2 + 7);
  a2[3] = (char *)-1LL;
  v6 = *(_BYTE **)(a1 + 16);
  v18 = v5;
  (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v6 + 32LL))(v6);
  if ( (v6[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v6 + 40LL))(v6, a2);
  v7 = *((_DWORD *)a2 + 7);
  if ( v7 == -1 )
  {
    sub_E12F20((__int64 *)a2, 3u, "...");
  }
  else if ( v7 )
  {
    for ( i = 1; v7 != i; ++i )
    {
      v9 = a2[1];
      v10 = (unsigned __int64)a2[2];
      v11 = (__int64)*a2;
      if ( (unsigned __int64)(v9 + 2) > v10 )
      {
        v12 = (unsigned __int64)(v9 + 994);
        v13 = 2 * v10;
        if ( v12 > v13 )
          a2[2] = (char *)v12;
        else
          a2[2] = (char *)v13;
        v14 = realloc((void *)v11);
        *a2 = (char *)v14;
        v11 = v14;
        if ( !v14 )
          abort();
        v9 = a2[1];
      }
      *(_WORD *)&v9[v11] = 8236;
      a2[1] += 2;
      *((_DWORD *)a2 + 6) = i;
      v15 = *(_BYTE **)(a1 + 16);
      (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v15 + 32LL))(v15, a2);
      if ( (v15[9] & 0xC0) != 0x40 )
        (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v15 + 40LL))(v15, a2);
    }
  }
  else
  {
    a2[1] = v4;
  }
  *((_DWORD *)a2 + 7) = v18;
  *((_DWORD *)a2 + 6) = v17;
  return v17;
}
