// Function: sub_B33910
// Address: 0xb33910
//
_QWORD *__fastcall sub_B33910(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx

  v2 = *a2;
  v3 = *a2 + 16LL * *((unsigned int *)a2 + 2);
  if ( v3 == *a2 )
  {
LABEL_6:
    *a1 = 0;
    return a1;
  }
  else
  {
    while ( *(_DWORD *)v2 )
    {
      v2 += 16;
      if ( v3 == v2 )
        goto LABEL_6;
    }
    sub_B10CB0(a1, *(_QWORD *)(v2 + 8));
    return a1;
  }
}
