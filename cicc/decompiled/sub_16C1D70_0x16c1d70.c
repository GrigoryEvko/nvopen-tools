// Function: sub_16C1D70
// Address: 0x16c1d70
//
void __fastcall sub_16C1D70(char *a1, __int64 a2)
{
  _BYTE *v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // r13
  int v6; // r14d
  unsigned int v7; // eax
  _BYTE *v8; // rsi
  _BYTE *v9; // rdi
  _BYTE *v10; // [rsp+0h] [rbp-50h] BYREF
  size_t n; // [rsp+8h] [rbp-48h]
  _BYTE src[64]; // [rsp+10h] [rbp-40h] BYREF

  sub_16C1C80(&v10, a1);
  v3 = v10;
  if ( v10 == src )
  {
    v4 = (unsigned int)n;
    v5 = *(unsigned int *)(a2 + 8);
    v6 = n;
    if ( (unsigned int)n <= v5 )
    {
      v9 = src;
      if ( (_DWORD)n )
      {
        memmove(*(void **)a2, src, (unsigned int)n);
        v9 = v10;
      }
    }
    else
    {
      if ( (unsigned int)n > (unsigned __int64)*(unsigned int *)(a2 + 12) )
      {
        v5 = 0;
        *(_DWORD *)(a2 + 8) = 0;
        sub_16CD150(a2, a2 + 16, v4, 1);
        v9 = v10;
        v4 = (unsigned int)n;
        v8 = v10;
        v7 = n;
      }
      else if ( *(_DWORD *)(a2 + 8) )
      {
        memmove(*(void **)a2, src, *(unsigned int *)(a2 + 8));
        v9 = v10;
        v4 = (unsigned int)n;
        v8 = &v10[v5];
        v7 = n;
      }
      else
      {
        v7 = n;
        v8 = src;
        v9 = src;
      }
      if ( v8 != &v9[v4] )
      {
        memcpy((void *)(v5 + *(_QWORD *)a2), v8, v7 - v5);
        v9 = v10;
      }
    }
    *(_DWORD *)(a2 + 8) = v6;
    if ( v9 != src )
      _libc_free((unsigned __int64)v9);
  }
  else
  {
    if ( *(_QWORD *)a2 != a2 + 16 )
    {
      _libc_free(*(_QWORD *)a2);
      v3 = v10;
    }
    *(_QWORD *)a2 = v3;
    *(_QWORD *)(a2 + 8) = n;
  }
}
