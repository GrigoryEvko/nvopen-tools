// Function: sub_1EE5780
// Address: 0x1ee5780
//
void __fastcall sub_1EE5780(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  char *v4; // rdi
  char *v5; // r8
  __int64 v6; // rcx
  __int64 v7; // rdx
  char *v8; // rdx
  bool v9; // zf
  __int64 v10; // rdx
  int v11; // eax

  v3 = *(unsigned int *)(a1 + 8);
  v4 = *(char **)a1;
  v3 *= 8;
  v5 = &v4[v3];
  v6 = v3 >> 3;
  v7 = v3 >> 5;
  if ( v7 )
  {
    v8 = &v4[32 * v7];
    while ( (_DWORD)a2 != *(_DWORD *)v4 )
    {
      if ( (_DWORD)a2 == *((_DWORD *)v4 + 2) )
      {
        v4 += 8;
        goto LABEL_8;
      }
      if ( (_DWORD)a2 == *((_DWORD *)v4 + 4) )
      {
        v4 += 16;
        goto LABEL_8;
      }
      if ( (_DWORD)a2 == *((_DWORD *)v4 + 6) )
      {
        v4 += 24;
        goto LABEL_8;
      }
      v4 += 32;
      if ( v8 == v4 )
      {
        v6 = (v5 - v4) >> 3;
        goto LABEL_15;
      }
    }
    goto LABEL_8;
  }
LABEL_15:
  if ( v6 == 2 )
  {
LABEL_22:
    if ( (_DWORD)a2 != *(_DWORD *)v4 )
    {
      v4 += 8;
      goto LABEL_18;
    }
    goto LABEL_8;
  }
  if ( v6 != 3 )
  {
    if ( v6 != 1 )
      return;
LABEL_18:
    if ( (_DWORD)a2 != *(_DWORD *)v4 )
      return;
    goto LABEL_8;
  }
  if ( (_DWORD)a2 != *(_DWORD *)v4 )
  {
    v4 += 8;
    goto LABEL_22;
  }
LABEL_8:
  if ( v5 != v4 )
  {
    v9 = (~HIDWORD(a2) & *((_DWORD *)v4 + 1)) == 0;
    *((_DWORD *)v4 + 1) &= ~HIDWORD(a2);
    if ( v9 )
    {
      v10 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
      v11 = *(_DWORD *)(a1 + 8);
      if ( (char *)v10 != v4 + 8 )
      {
        memmove(v4, v4 + 8, v10 - (_QWORD)(v4 + 8));
        v11 = *(_DWORD *)(a1 + 8);
      }
      *(_DWORD *)(a1 + 8) = v11 - 1;
    }
  }
}
