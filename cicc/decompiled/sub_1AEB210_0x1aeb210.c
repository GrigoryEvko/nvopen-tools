// Function: sub_1AEB210
// Address: 0x1aeb210
//
__int64 __fastcall sub_1AEB210(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // r15
  __int64 *v4; // rbx
  __int64 *v5; // r14
  __int64 v6; // r12
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax

  for ( result = *(unsigned int *)(a1 + 8); (_DWORD)result; result = *(unsigned int *)(a1 + 8) )
  {
    v3 = *(_QWORD *)(*(_QWORD *)a1 + 8LL * (unsigned int)result - 8);
    *(_DWORD *)(a1 + 8) = result - 1;
    sub_1AEAA40(v3);
    if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
    {
      v4 = *(__int64 **)(v3 - 8);
      v5 = &v4[3 * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF)];
    }
    else
    {
      v5 = (__int64 *)v3;
      v4 = (__int64 *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
    }
    for ( ; v5 != v4; ++*(_DWORD *)(a1 + 8) )
    {
      while ( 1 )
      {
        v6 = *v4;
        if ( *v4 )
        {
          v7 = v4[1];
          v8 = v4[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v8 = v7;
          if ( v7 )
            *(_QWORD *)(v7 + 16) = *(_QWORD *)(v7 + 16) & 3LL | v8;
        }
        *v4 = 0;
        if ( !*(_QWORD *)(v6 + 8) && *(_BYTE *)(v6 + 16) > 0x17u && sub_1AE9990(v6, a2) )
          break;
        v4 += 3;
        if ( v5 == v4 )
          goto LABEL_16;
      }
      v11 = *(unsigned int *)(a1 + 8);
      if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 12) )
      {
        sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, v9, v10);
        v11 = *(unsigned int *)(a1 + 8);
      }
      v4 += 3;
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v11) = v6;
    }
LABEL_16:
    sub_15F20C0((_QWORD *)v3);
  }
  return result;
}
