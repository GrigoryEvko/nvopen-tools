// Function: sub_1983E30
// Address: 0x1983e30
//
__int64 __fastcall sub_1983E30(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // r14
  __int64 v4; // r13
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 v8; // rdx
  _QWORD *v9; // rdx

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v3 = *(_QWORD **)(a2 + 72);
    do
    {
      v4 = sub_1648700(v2)[5];
      v5 = *(_QWORD **)(a2 + 64);
      if ( v3 == v5 )
      {
        v8 = *(unsigned int *)(a2 + 84);
        v6 = &v3[v8];
        if ( v3 == v6 )
        {
          v9 = v3;
        }
        else
        {
          do
          {
            if ( v4 == *v5 )
              break;
            ++v5;
          }
          while ( v6 != v5 );
          v9 = &v3[v8];
        }
      }
      else
      {
        v6 = &v3[*(unsigned int *)(a2 + 80)];
        v5 = sub_16CC9F0(a2 + 56, v4);
        if ( v4 == *v5 )
        {
          v3 = *(_QWORD **)(a2 + 72);
          if ( v3 == *(_QWORD **)(a2 + 64) )
            v9 = &v3[*(unsigned int *)(a2 + 84)];
          else
            v9 = &v3[*(unsigned int *)(a2 + 80)];
        }
        else
        {
          v3 = *(_QWORD **)(a2 + 72);
          if ( v3 != *(_QWORD **)(a2 + 64) )
          {
            v5 = &v3[*(unsigned int *)(a2 + 80)];
            goto LABEL_8;
          }
          v5 = &v3[*(unsigned int *)(a2 + 84)];
          v9 = v5;
        }
      }
      while ( v9 != v5 && *v5 >= 0xFFFFFFFFFFFFFFFELL )
        ++v5;
LABEL_8:
      if ( v5 == v6 )
        return 1;
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v2 );
  }
  return 0;
}
