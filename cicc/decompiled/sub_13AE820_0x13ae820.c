// Function: sub_13AE820
// Address: 0x13ae820
//
unsigned __int64 __fastcall sub_13AE820(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned __int64 *v5; // r12
  unsigned __int64 *v6; // r12
  unsigned __int64 *v7; // r12
  __int64 i; // rdx

  result = *(unsigned int *)(a1 + 8);
  if ( a2 < result )
  {
    v3 = *(_QWORD *)a1 + 48 * result;
    v4 = *(_QWORD *)a1 + 48 * a2;
    while ( v4 != v3 )
    {
      while ( 1 )
      {
        v5 = *(unsigned __int64 **)(v3 - 8);
        v3 -= 48;
        if ( ((unsigned __int8)v5 & 1) == 0 && v5 )
        {
          _libc_free(*v5);
          result = j_j___libc_free_0(v5, 24);
        }
        v6 = *(unsigned __int64 **)(v3 + 32);
        if ( ((unsigned __int8)v6 & 1) == 0 && v6 )
        {
          _libc_free(*v6);
          result = j_j___libc_free_0(v6, 24);
        }
        v7 = *(unsigned __int64 **)(v3 + 24);
        if ( ((unsigned __int8)v7 & 1) != 0 || !v7 )
          break;
        _libc_free(*v7);
        result = j_j___libc_free_0(v7, 24);
        if ( v4 == v3 )
          goto LABEL_13;
      }
    }
    goto LABEL_13;
  }
  if ( a2 > result )
  {
    if ( a2 > *(unsigned int *)(a1 + 12) )
    {
      sub_13AE5E0(a1, a2);
      result = *(unsigned int *)(a1 + 8);
    }
    result = *(_QWORD *)a1 + 48 * result;
    for ( i = *(_QWORD *)a1 + 48 * a2; i != result; result += 48LL )
    {
      if ( result )
      {
        *(_QWORD *)result = 0;
        *(_QWORD *)(result + 8) = 0;
        *(_DWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 1;
        *(_QWORD *)(result + 32) = 1;
        *(_QWORD *)(result + 40) = 1;
      }
    }
LABEL_13:
    *(_DWORD *)(a1 + 8) = a2;
  }
  return result;
}
