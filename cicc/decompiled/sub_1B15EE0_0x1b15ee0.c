// Function: sub_1B15EE0
// Address: 0x1b15ee0
//
__int64 __fastcall sub_1B15EE0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rax
  __int64 *v4; // r12
  _QWORD *v5; // rcx
  __int64 v6; // r13
  _QWORD *v7; // rax
  _QWORD *v8; // r14
  __int64 v10; // rdx
  _QWORD *v11; // rdx

  v2 = (__int64 *)a1;
  v3 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v4 = *(__int64 **)(a1 - 8);
    v2 = &v4[v3];
  }
  else
  {
    v4 = (__int64 *)(a1 - v3 * 8);
  }
  if ( v2 != v4 )
  {
    v5 = *(_QWORD **)(a2 + 16);
    do
    {
      v6 = *v4;
      if ( *(_BYTE *)(*v4 + 16) <= 0x17u )
        v6 = 0;
      v7 = *(_QWORD **)(a2 + 8);
      if ( v5 == v7 )
      {
        v10 = *(unsigned int *)(a2 + 28);
        v8 = &v5[v10];
        if ( v5 == v8 )
        {
          v11 = v5;
        }
        else
        {
          do
          {
            if ( v6 == *v7 )
              break;
            ++v7;
          }
          while ( v8 != v7 );
          v11 = &v5[v10];
        }
      }
      else
      {
        v8 = &v5[*(unsigned int *)(a2 + 24)];
        v7 = sub_16CC9F0(a2, v6);
        if ( v6 == *v7 )
        {
          v5 = *(_QWORD **)(a2 + 16);
          if ( v5 == *(_QWORD **)(a2 + 8) )
            v11 = &v5[*(unsigned int *)(a2 + 28)];
          else
            v11 = &v5[*(unsigned int *)(a2 + 24)];
        }
        else
        {
          v5 = *(_QWORD **)(a2 + 16);
          if ( v5 != *(_QWORD **)(a2 + 8) )
          {
            v7 = &v5[*(unsigned int *)(a2 + 24)];
            goto LABEL_12;
          }
          v7 = &v5[*(unsigned int *)(a2 + 28)];
          v11 = v7;
        }
      }
      while ( v11 != v7 && *v7 >= 0xFFFFFFFFFFFFFFFELL )
        ++v7;
LABEL_12:
      if ( v7 == v8 )
        return 0;
      v4 += 3;
    }
    while ( v2 != v4 );
  }
  return 1;
}
