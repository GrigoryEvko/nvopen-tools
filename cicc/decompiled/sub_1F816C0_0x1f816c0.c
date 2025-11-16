// Function: sub_1F816C0
// Address: 0x1f816c0
//
_BOOL8 __fastcall sub_1F816C0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rdx
  _QWORD *v3; // rax
  _QWORD *v4; // r13
  __int64 v5; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // rdx

  v2 = *(_QWORD **)(a1 + 16);
  v3 = *(_QWORD **)(a1 + 8);
  if ( v2 == v3 )
  {
    v4 = &v3[*(unsigned int *)(a1 + 28)];
    if ( v3 == v4 )
    {
      v9 = *(_QWORD **)(a1 + 8);
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
    v4 = &v2[*(unsigned int *)(a1 + 24)];
    v3 = sub_16CC9F0(a1, a2);
    if ( a2 == *v3 )
    {
      v7 = *(_QWORD *)(a1 + 16);
      if ( v7 == *(_QWORD *)(a1 + 8) )
        v8 = *(unsigned int *)(a1 + 28);
      else
        v8 = *(unsigned int *)(a1 + 24);
      v9 = (_QWORD *)(v7 + 8 * v8);
    }
    else
    {
      v5 = *(_QWORD *)(a1 + 16);
      if ( v5 != *(_QWORD *)(a1 + 8) )
      {
        v3 = (_QWORD *)(v5 + 8LL * *(unsigned int *)(a1 + 24));
        return v3 != v4;
      }
      v3 = (_QWORD *)(v5 + 8LL * *(unsigned int *)(a1 + 28));
      v9 = v3;
    }
  }
  for ( ; v9 != v3; ++v3 )
  {
    if ( *v3 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  return v3 != v4;
}
