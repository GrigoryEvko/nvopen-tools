// Function: sub_1DE89B0
// Address: 0x1de89b0
//
__int64 __fastcall sub_1DE89B0(__int64 a1, __int64 a2)
{
  __int64 *v3; // r12
  __int64 *v4; // r15
  _QWORD *v5; // rcx
  _QWORD *v6; // r13
  _QWORD *v7; // rax
  __int64 v8; // r14
  __int64 v9; // rdx
  _QWORD *v10; // rdx

  if ( *(_DWORD *)(a2 + 28) - *(_DWORD *)(a2 + 32) != (unsigned int)((__int64)(*(_QWORD *)(a1 + 96)
                                                                             - *(_QWORD *)(a1 + 88)) >> 3) )
    return 0;
  if ( sub_1DA1810(a2, a1) )
    return 0;
  v3 = *(__int64 **)(a1 + 96);
  v4 = *(__int64 **)(a1 + 88);
  if ( v3 != v4 )
  {
    v5 = *(_QWORD **)(a2 + 16);
    do
    {
      v7 = *(_QWORD **)(a2 + 8);
      v8 = *v4;
      if ( v5 == v7 )
      {
        v9 = *(unsigned int *)(a2 + 28);
        v6 = &v5[v9];
        if ( v5 == v6 )
        {
          v10 = v5;
        }
        else
        {
          do
          {
            if ( v8 == *v7 )
              break;
            ++v7;
          }
          while ( v6 != v7 );
          v10 = &v5[v9];
        }
LABEL_19:
        while ( v10 != v7 )
        {
          if ( *v7 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_9;
          ++v7;
        }
        if ( v6 == v7 )
          return 0;
      }
      else
      {
        v6 = &v5[*(unsigned int *)(a2 + 24)];
        v7 = sub_16CC9F0(a2, *v4);
        if ( v8 == *v7 )
        {
          v5 = *(_QWORD **)(a2 + 16);
          if ( v5 == *(_QWORD **)(a2 + 8) )
            v10 = &v5[*(unsigned int *)(a2 + 28)];
          else
            v10 = &v5[*(unsigned int *)(a2 + 24)];
          goto LABEL_19;
        }
        v5 = *(_QWORD **)(a2 + 16);
        if ( v5 == *(_QWORD **)(a2 + 8) )
        {
          v7 = &v5[*(unsigned int *)(a2 + 28)];
          v10 = v7;
          goto LABEL_19;
        }
        v7 = &v5[*(unsigned int *)(a2 + 24)];
LABEL_9:
        if ( v6 == v7 )
          return 0;
      }
      ++v4;
    }
    while ( v3 != v4 );
  }
  return 1;
}
