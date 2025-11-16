// Function: sub_2E79700
// Address: 0x2e79700
//
unsigned __int64 __fastcall sub_2E79700(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r12
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 result; // rax
  __int64 v7; // rsi
  unsigned int v8; // edx
  __int64 *v9; // rcx
  __int64 v10; // rdi
  int v11; // ecx
  int v12; // r9d
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v14; // [rsp+10h] [rbp-30h]

  v2 = a2;
  if ( *(_WORD *)(a2 + 68) == 21 )
    v2 = sub_2E78040(a2);
  sub_2E79610(&v13, a1, v2);
  v4 = v14;
  if ( v14 != (_QWORD *)(*(_QWORD *)(a1 + 696) + 32LL * *(unsigned int *)(a1 + 712)) )
  {
    v5 = v14[1];
    if ( (_QWORD *)v5 != v14 + 3 )
      _libc_free(v5);
    *v4 = -8192;
    --*(_DWORD *)(a1 + 704);
    ++*(_DWORD *)(a1 + 708);
  }
  result = *(unsigned int *)(a1 + 744);
  v7 = *(_QWORD *)(a1 + 728);
  if ( (_DWORD)result )
  {
    v8 = (result - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
    v9 = (__int64 *)(v7 + 24LL * v8);
    v10 = *v9;
    if ( v2 == *v9 )
    {
LABEL_9:
      result = v7 + 24 * result;
      if ( v9 != (__int64 *)result )
      {
        *v9 = -8192;
        --*(_DWORD *)(a1 + 736);
        ++*(_DWORD *)(a1 + 740);
      }
    }
    else
    {
      v11 = 1;
      while ( v10 != -4096 )
      {
        v12 = v11 + 1;
        v8 = (result - 1) & (v11 + v8);
        v9 = (__int64 *)(v7 + 24LL * v8);
        v10 = *v9;
        if ( v2 == *v9 )
          goto LABEL_9;
        v11 = v12;
      }
    }
  }
  return result;
}
