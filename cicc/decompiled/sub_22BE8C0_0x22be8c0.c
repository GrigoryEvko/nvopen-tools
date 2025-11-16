// Function: sub_22BE8C0
// Address: 0x22be8c0
//
__int64 __fastcall sub_22BE8C0(unsigned __int8 *a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  int v4; // eax
  int v5; // eax
  int v6; // eax
  int v7; // eax
  unsigned __int64 v8; // rdi

  if ( !a1[40] )
  {
    result = *a2;
    *(_WORD *)a1 = (unsigned __int8)result;
    if ( (unsigned __int8)result > 3u )
    {
      result = (unsigned int)(result - 4);
      if ( (unsigned __int8)result <= 1u )
      {
        v6 = *((_DWORD *)a2 + 4);
        *((_DWORD *)a2 + 4) = 0;
        *((_DWORD *)a1 + 4) = v6;
        *((_QWORD *)a1 + 1) = *((_QWORD *)a2 + 1);
        v7 = *((_DWORD *)a2 + 8);
        *((_DWORD *)a2 + 8) = 0;
        *((_DWORD *)a1 + 8) = v7;
        *((_QWORD *)a1 + 3) = *((_QWORD *)a2 + 3);
        result = a2[1];
        *a2 = 0;
        a1[1] = result;
        a1[40] = 1;
        return result;
      }
    }
    else if ( (unsigned __int8)result > 1u )
    {
      result = *((_QWORD *)a2 + 1);
      *((_QWORD *)a1 + 1) = result;
    }
    *a2 = 0;
    a1[40] = 1;
    return result;
  }
  if ( (unsigned int)*a1 - 4 <= 1 )
  {
    if ( *((_DWORD *)a1 + 8) > 0x40u )
    {
      v8 = *((_QWORD *)a1 + 3);
      if ( v8 )
        j_j___libc_free_0_0(v8);
    }
    sub_969240((__int64 *)a1 + 1);
  }
  result = *a2;
  *(_WORD *)a1 = (unsigned __int8)result;
  if ( (unsigned __int8)result > 3u )
  {
    result = (unsigned int)(result - 4);
    if ( (unsigned __int8)result <= 1u )
    {
      v4 = *((_DWORD *)a2 + 4);
      *((_DWORD *)a2 + 4) = 0;
      *((_DWORD *)a1 + 4) = v4;
      *((_QWORD *)a1 + 1) = *((_QWORD *)a2 + 1);
      v5 = *((_DWORD *)a2 + 8);
      *((_DWORD *)a2 + 8) = 0;
      *((_DWORD *)a1 + 8) = v5;
      *((_QWORD *)a1 + 3) = *((_QWORD *)a2 + 3);
      result = a2[1];
      a1[1] = result;
    }
  }
  else if ( (unsigned __int8)result > 1u )
  {
    result = *((_QWORD *)a2 + 1);
    *((_QWORD *)a1 + 1) = result;
  }
  *a2 = 0;
  return result;
}
