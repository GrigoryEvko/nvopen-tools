// Function: sub_1607100
// Address: 0x1607100
//
_QWORD *__fastcall sub_1607100(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // rax
  __int64 v4; // rdi
  _QWORD *v5; // rcx

  v2 = (_QWORD *)a2[1];
  v4 = *a2;
  v5 = &v2[*((unsigned int *)a2 + 6)];
  if ( *((_DWORD *)a2 + 4) )
  {
    for ( ; v2 != v5; ++v2 )
    {
      if ( *v2 != -8 && *v2 != -16 )
        break;
    }
  }
  else
  {
    v2 += *((unsigned int *)a2 + 6);
  }
  a1[2] = v2;
  *a1 = a2;
  a1[1] = v4;
  a1[3] = v5;
  return a1;
}
