// Function: sub_13EABD0
// Address: 0x13eabd0
//
__int64 __fastcall sub_13EABD0(unsigned int *a1, __int64 a2)
{
  unsigned int v3; // eax
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rdi

  if ( *a1 != 3 )
  {
    if ( !(unsigned __int8)sub_158A120(a2) )
    {
      *a1 = 3;
      v3 = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a2 + 8) = 0;
      a1[4] = v3;
      *((_QWORD *)a1 + 1) = *(_QWORD *)a2;
      a1[8] = *(_DWORD *)(a2 + 24);
      result = *(_QWORD *)(a2 + 16);
      *((_QWORD *)a1 + 3) = result;
      *(_DWORD *)(a2 + 24) = 0;
      return result;
    }
    goto LABEL_6;
  }
  if ( (unsigned __int8)sub_158A120(a2) )
  {
LABEL_6:
    result = *a1;
    if ( (_DWORD)result != 4 )
    {
      if ( (unsigned int)(result - 1) > 1 )
      {
        if ( (_DWORD)result == 3 )
        {
          if ( a1[8] > 0x40 )
          {
            v5 = *((_QWORD *)a1 + 3);
            if ( v5 )
              result = j_j___libc_free_0_0(v5);
          }
          if ( a1[4] > 0x40 )
          {
            v6 = *((_QWORD *)a1 + 1);
            if ( v6 )
              result = j_j___libc_free_0_0(v6);
          }
        }
      }
      else
      {
        *((_QWORD *)a1 + 1) = 0;
      }
      *a1 = 4;
    }
    return result;
  }
  if ( a1[4] > 0x40 )
  {
    v7 = *((_QWORD *)a1 + 1);
    if ( v7 )
      j_j___libc_free_0_0(v7);
  }
  *((_QWORD *)a1 + 1) = *(_QWORD *)a2;
  a1[4] = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 8) = 0;
  if ( a1[8] > 0x40 )
  {
    v8 = *((_QWORD *)a1 + 3);
    if ( v8 )
      j_j___libc_free_0_0(v8);
  }
  *((_QWORD *)a1 + 3) = *(_QWORD *)(a2 + 16);
  result = *(unsigned int *)(a2 + 24);
  a1[8] = result;
  *(_DWORD *)(a2 + 24) = 0;
  return result;
}
