// Function: sub_3507B80
// Address: 0x3507b80
//
__int64 __fastcall sub_3507B80(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  __int64 result; // rax
  __int16 *v8; // r13
  unsigned __int16 v10; // bx
  const void *v11; // r15
  _BYTE *v12; // rdi
  __int64 v13; // r8
  unsigned int v14; // eax
  __int64 v15; // rcx
  _WORD *v16; // rdx
  __int64 v17; // rax

  v6 = a2;
  result = *(_QWORD *)(*a1 + 56LL);
  v8 = (__int16 *)(result + 2LL * *(unsigned int *)(*(_QWORD *)(*a1 + 8LL) + 24LL * a2 + 4));
  if ( v8 )
  {
    v10 = a2;
    v11 = a1 + 4;
    while ( 1 )
    {
      v12 = (_BYTE *)(a1[6] + v10);
      v13 = a1[2];
      v14 = (unsigned __int8)*v12;
      if ( v14 >= (unsigned int)v13 )
        goto LABEL_11;
      v15 = a1[1];
      while ( 1 )
      {
        v16 = (_WORD *)(v15 + 2LL * v14);
        if ( *v16 == v10 )
          break;
        v14 += 256;
        if ( (unsigned int)v13 <= v14 )
          goto LABEL_11;
      }
      if ( v16 == (_WORD *)(v15 + 2 * v13) )
      {
LABEL_11:
        *v12 = v13;
        v17 = a1[2];
        if ( (unsigned __int64)(v17 + 1) > a1[3] )
        {
          sub_C8D290((__int64)(a1 + 1), v11, v17 + 1, 2u, v13, a6);
          v17 = a1[2];
        }
        *(_WORD *)(a1[1] + 2 * v17) = v10;
        ++a1[2];
      }
      result = (unsigned int)*v8++;
      if ( !(_WORD)result )
        break;
      v6 += result;
      v10 = v6;
    }
  }
  return result;
}
