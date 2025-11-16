// Function: sub_3959390
// Address: 0x3959390
//
__int64 __fastcall sub_3959390(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 result; // rax
  __int64 v4; // r13
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx

  v2 = *(__int64 **)a2;
  result = 3LL * *(unsigned int *)(a2 + 8);
  v4 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( v4 != *(_QWORD *)a2 )
  {
    do
    {
      v6 = *(_QWORD **)a1;
      v7 = *(unsigned int *)(a1 + 8);
      result = 3 * v7;
      v8 = *(_QWORD *)a1 + 24 * v7;
      if ( *(_QWORD *)a1 != v8 )
      {
        result = *v2;
        while ( result != *v6 || v2[1] != v6[1] || v2[2] != v6[2] )
        {
          v6 += 3;
          if ( v6 == (_QWORD *)v8 )
            goto LABEL_11;
        }
        if ( (_QWORD *)v8 != v6 + 3 )
        {
          result = (__int64)memmove(v6, v6 + 3, v8 - (_QWORD)(v6 + 3));
          LODWORD(v7) = *(_DWORD *)(a1 + 8);
        }
        *(_DWORD *)(a1 + 8) = v7 - 1;
      }
LABEL_11:
      v2 += 3;
    }
    while ( (__int64 *)v4 != v2 );
  }
  return result;
}
