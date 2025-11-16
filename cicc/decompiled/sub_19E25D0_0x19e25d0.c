// Function: sub_19E25D0
// Address: 0x19e25d0
//
void __fastcall sub_19E25D0(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  _QWORD *v3; // rdx
  _QWORD *v4; // rax
  _QWORD *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // rdx

  v1 = *a1;
  if ( *a1 != a1[1] )
  {
    while ( 1 )
    {
      v2 = a1[2];
      if ( v1 )
        v1 -= 24;
      v3 = *(_QWORD **)(v2 + 2248);
      v4 = *(_QWORD **)(v2 + 2240);
      if ( v3 == v4 )
      {
        v5 = &v4[*(unsigned int *)(v2 + 2260)];
        if ( v4 == v5 )
        {
          v9 = *(_QWORD **)(v2 + 2240);
        }
        else
        {
          do
          {
            if ( v1 == *v4 )
              break;
            ++v4;
          }
          while ( v5 != v4 );
          v9 = v5;
        }
        goto LABEL_19;
      }
      v5 = &v3[*(unsigned int *)(v2 + 2256)];
      v4 = sub_16CC9F0(v2 + 2232, v1);
      if ( v1 == *v4 )
        break;
      v6 = *(_QWORD *)(v2 + 2248);
      if ( v6 == *(_QWORD *)(v2 + 2240) )
      {
        v4 = (_QWORD *)(v6 + 8LL * *(unsigned int *)(v2 + 2260));
        v9 = v4;
LABEL_19:
        while ( v9 != v4 && *v4 >= 0xFFFFFFFFFFFFFFFELL )
          ++v4;
        goto LABEL_10;
      }
      v4 = (_QWORD *)(v6 + 8LL * *(unsigned int *)(v2 + 2256));
LABEL_10:
      if ( v4 != v5 )
      {
        v1 = *(_QWORD *)(*a1 + 8);
        *a1 = v1;
        if ( a1[1] != v1 )
          continue;
      }
      return;
    }
    v7 = *(_QWORD *)(v2 + 2248);
    if ( v7 == *(_QWORD *)(v2 + 2240) )
      v8 = *(unsigned int *)(v2 + 2260);
    else
      v8 = *(unsigned int *)(v2 + 2256);
    v9 = (_QWORD *)(v7 + 8 * v8);
    goto LABEL_19;
  }
}
