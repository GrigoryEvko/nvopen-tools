// Function: sub_1BF2790
// Address: 0x1bf2790
//
bool __fastcall sub_1BF2790(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  _QWORD *v3; // rax
  _QWORD *v4; // r13
  __int64 v5; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // rdx

  if ( *(_BYTE *)(a2 + 16) > 0x17u )
  {
    v2 = *(_QWORD **)(a1 + 176);
    v3 = *(_QWORD **)(a1 + 168);
    if ( v2 == v3 )
    {
      v4 = &v3[*(unsigned int *)(a1 + 188)];
      if ( v3 == v4 )
      {
        v9 = *(_QWORD **)(a1 + 168);
      }
      else
      {
        do
        {
          if ( a2 == *v3 )
            break;
          ++v3;
        }
        while ( v4 != v3 );
        v9 = v4;
      }
    }
    else
    {
      v4 = &v2[*(unsigned int *)(a1 + 184)];
      v3 = sub_16CC9F0(a1 + 160, a2);
      if ( a2 == *v3 )
      {
        v7 = *(_QWORD *)(a1 + 176);
        if ( v7 == *(_QWORD *)(a1 + 168) )
          v8 = *(unsigned int *)(a1 + 188);
        else
          v8 = *(unsigned int *)(a1 + 184);
        v9 = (_QWORD *)(v7 + 8 * v8);
      }
      else
      {
        v5 = *(_QWORD *)(a1 + 176);
        if ( v5 != *(_QWORD *)(a1 + 168) )
        {
          v3 = (_QWORD *)(v5 + 8LL * *(unsigned int *)(a1 + 184));
          return v4 != v3;
        }
        v3 = (_QWORD *)(v5 + 8LL * *(unsigned int *)(a1 + 188));
        v9 = v3;
      }
    }
    while ( v9 != v3 && *v3 >= 0xFFFFFFFFFFFFFFFELL )
      ++v3;
    return v4 != v3;
  }
  return 0;
}
