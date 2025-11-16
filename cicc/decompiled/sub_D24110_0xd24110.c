// Function: sub_D24110
// Address: 0xd24110
//
_QWORD **__fastcall sub_D24110(__int64 a1)
{
  _QWORD **result; // rax
  _QWORD **i; // rcx
  _QWORD *v3; // rdx
  _QWORD *v4; // rdx
  _QWORD *v5; // rcx
  _QWORD *v6; // rax
  _QWORD *v7; // rdx

  if ( *(_DWORD *)(a1 + 112) )
  {
    v4 = *(_QWORD **)(a1 + 104);
    v5 = &v4[2 * *(unsigned int *)(a1 + 120)];
    if ( v4 != v5 )
    {
      while ( 1 )
      {
        v6 = v4;
        if ( *v4 != -4096 && *v4 != -8192 )
          break;
        v4 += 2;
        if ( v5 == v4 )
          goto LABEL_2;
      }
      while ( v5 != v6 )
      {
        v7 = (_QWORD *)v6[1];
        v6 += 2;
        *v7 = a1;
        if ( v6 == v5 )
          break;
        while ( *v6 == -8192 || *v6 == -4096 )
        {
          v6 += 2;
          if ( v5 == v6 )
            goto LABEL_2;
        }
      }
    }
  }
LABEL_2:
  result = *(_QWORD ***)(a1 + 432);
  for ( i = &result[*(unsigned int *)(a1 + 440)]; i != result; *v3 = a1 )
    v3 = *result++;
  return result;
}
