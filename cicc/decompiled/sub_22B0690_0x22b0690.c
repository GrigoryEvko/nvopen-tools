// Function: sub_22B0690
// Address: 0x22b0690
//
_QWORD *__fastcall sub_22B0690(_QWORD *a1, __int64 *a2)
{
  _DWORD *v2; // rax
  __int64 v3; // rcx
  _DWORD *v4; // rdx

  v2 = (_DWORD *)a2[1];
  v3 = *a2;
  v4 = &v2[*((unsigned int *)a2 + 6)];
  if ( *((_DWORD *)a2 + 4) )
  {
    for ( ; v2 != v4; ++v2 )
    {
      if ( *v2 <= 0xFFFFFFFD )
        break;
    }
  }
  else
  {
    v2 += *((unsigned int *)a2 + 6);
  }
  a1[2] = v2;
  *a1 = a2;
  a1[1] = v3;
  a1[3] = v4;
  return a1;
}
