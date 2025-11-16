// Function: sub_1DC1BF0
// Address: 0x1dc1bf0
//
__int64 __fastcall sub_1DC1BF0(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rdx
  const void *v7; // r15
  unsigned __int16 v8; // r14
  __int64 result; // rax
  unsigned __int16 *v11; // rdx
  unsigned __int16 *v12; // r12
  _BYTE *v13; // rdi
  unsigned int v14; // ecx
  unsigned int v15; // eax
  __int64 v16; // rsi
  _DWORD *v17; // rdx
  __int64 v18; // rax

  v6 = *a1;
  if ( !*a1 )
    BUG();
  v7 = a1 + 3;
  v8 = a2;
  result = *(_QWORD *)(v6 + 56);
  v11 = (unsigned __int16 *)(result + 2LL * *(unsigned int *)(*(_QWORD *)(v6 + 8) + 24LL * a2 + 4));
  while ( 1 )
  {
    v12 = v11;
    if ( !v11 )
      return result;
    while ( 1 )
    {
      v13 = (_BYTE *)(a1[7] + v8);
      v14 = *((_DWORD *)a1 + 4);
      v15 = (unsigned __int8)*v13;
      if ( v15 >= v14 )
        goto LABEL_12;
      v16 = a1[1];
      while ( 1 )
      {
        v17 = (_DWORD *)(v16 + 4LL * v15);
        if ( v8 == *v17 )
          break;
        v15 += 256;
        if ( v14 <= v15 )
          goto LABEL_12;
      }
      if ( v17 == (_DWORD *)(v16 + 4LL * v14) )
      {
LABEL_12:
        *v13 = v14;
        v18 = *((unsigned int *)a1 + 4);
        if ( (unsigned int)v18 >= *((_DWORD *)a1 + 5) )
        {
          sub_16CD150((__int64)(a1 + 1), v7, 0, 4, a5, a6);
          v18 = *((unsigned int *)a1 + 4);
        }
        *(_DWORD *)(a1[1] + 4 * v18) = v8;
        ++*((_DWORD *)a1 + 4);
      }
      result = *v12;
      v11 = 0;
      ++v12;
      if ( !(_WORD)result )
        break;
      v8 += result;
      if ( !v12 )
        return result;
    }
  }
}
