// Function: sub_1C8D9F0
// Address: 0x1c8d9f0
//
bool __fastcall sub_1C8D9F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // rdx
  _QWORD *v5; // rax
  _QWORD *v6; // r13
  __int64 v7; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // rdx

  v2 = *a1;
  v3 = sub_1648700(a2);
  v4 = *(_QWORD **)(v2 + 16);
  v5 = *(_QWORD **)(v2 + 8);
  if ( v4 == v5 )
  {
    v6 = &v5[*(unsigned int *)(v2 + 28)];
    if ( v5 == v6 )
    {
      v11 = *(_QWORD **)(v2 + 8);
    }
    else
    {
      do
      {
        if ( v3 == (_QWORD *)*v5 )
          break;
        ++v5;
      }
      while ( v6 != v5 );
      v11 = v6;
    }
  }
  else
  {
    v6 = &v4[*(unsigned int *)(v2 + 24)];
    v5 = sub_16CC9F0(v2, (__int64)v3);
    if ( v3 == (_QWORD *)*v5 )
    {
      v9 = *(_QWORD *)(v2 + 16);
      if ( v9 == *(_QWORD *)(v2 + 8) )
        v10 = *(unsigned int *)(v2 + 28);
      else
        v10 = *(unsigned int *)(v2 + 24);
      v11 = (_QWORD *)(v9 + 8 * v10);
    }
    else
    {
      v7 = *(_QWORD *)(v2 + 16);
      if ( v7 != *(_QWORD *)(v2 + 8) )
      {
        v5 = (_QWORD *)(v7 + 8LL * *(unsigned int *)(v2 + 24));
        return v5 == v6;
      }
      v5 = (_QWORD *)(v7 + 8LL * *(unsigned int *)(v2 + 28));
      v11 = v5;
    }
  }
  for ( ; v11 != v5; ++v5 )
  {
    if ( *v5 < 0xFFFFFFFFFFFFFFFELL )
      break;
  }
  return v5 == v6;
}
