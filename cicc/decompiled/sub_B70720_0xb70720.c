// Function: sub_B70720
// Address: 0xb70720
//
__int64 __fastcall sub_B70720(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 *v2; // r12
  __int64 result; // rax
  _QWORD *v4; // r13
  _QWORD *v5; // rdi
  _QWORD *v6; // rdi

  v1 = *(__int64 **)(a1 + 8);
  v2 = &v1[*(unsigned int *)(a1 + 24)];
  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result && v1 != v2 )
  {
    while ( 1 )
    {
      result = *v1;
      if ( *v1 != -4096 && result != -8192 )
        break;
      if ( ++v1 == v2 )
        return result;
    }
LABEL_8:
    if ( v1 != v2 )
    {
      v4 = (_QWORD *)*v1;
      if ( *v1 )
      {
        v5 = (_QWORD *)v4[7];
        if ( v5 != v4 + 9 )
          j_j___libc_free_0(v5, v4[9] + 1LL);
        v6 = (_QWORD *)v4[3];
        if ( v6 != v4 + 5 )
          j_j___libc_free_0(v6, v4[5] + 1LL);
        sub_BD7260(v4);
        result = j_j___libc_free_0(v4, 112);
      }
      while ( ++v1 != v2 )
      {
        result = *v1;
        if ( *v1 != -4096 && result != -8192 )
          goto LABEL_8;
      }
    }
  }
  return result;
}
