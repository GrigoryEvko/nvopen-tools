// Function: sub_16060D0
// Address: 0x16060d0
//
__int64 __fastcall sub_16060D0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rdi

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v3 = v2 + 24 * result;
    while ( 1 )
    {
      result = *(unsigned int *)(v2 + 8);
      if ( (_DWORD)result )
        break;
      if ( *(_QWORD *)v2 <= 1u )
      {
LABEL_11:
        v2 += 24;
        if ( v3 == v2 )
          return result;
      }
      else
      {
        v4 = *(_QWORD *)(v2 + 16);
        if ( v4 )
          goto LABEL_4;
        v2 += 24;
        if ( v3 == v2 )
          return result;
      }
    }
    v4 = *(_QWORD *)(v2 + 16);
    if ( v4 )
    {
LABEL_4:
      if ( *(_DWORD *)(v4 + 32) > 0x40u )
      {
        v5 = *(_QWORD *)(v4 + 24);
        if ( v5 )
          j_j___libc_free_0_0(v5);
      }
      sub_164BE60(v4);
      sub_1648B90(v4);
      result = *(unsigned int *)(v2 + 8);
    }
    if ( (unsigned int)result > 0x40 )
    {
      if ( *(_QWORD *)v2 )
        result = j_j___libc_free_0_0(*(_QWORD *)v2);
    }
    goto LABEL_11;
  }
  return result;
}
