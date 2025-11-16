// Function: sub_33E0010
// Address: 0x33e0010
//
__int64 __fastcall sub_33E0010(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  int v6; // edx
  __int64 *v7; // rax
  int v8; // r12d
  __int64 *v9; // rax
  __int64 result; // rax
  __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rcx

  v4 = a2;
  v5 = a1;
  v6 = *(_DWORD *)(a1 + 24);
  if ( ((v6 - 214) & 0xFFFFFFFD) == 0 )
  {
    v7 = *(__int64 **)(a1 + 40);
    v5 = *v7;
    v6 = *(_DWORD *)(*v7 + 24);
  }
  v8 = a3;
  if ( ((*(_DWORD *)(a2 + 24) - 214) & 0xFFFFFFFD) == 0 )
  {
    v9 = *(__int64 **)(a2 + 40);
    v4 = *v9;
    v8 = *((_DWORD *)v9 + 2);
  }
  result = 0;
  if ( v6 == 186 )
  {
    v11 = sub_33DFEB0(
            **(_QWORD **)(v5 + 40),
            *(_QWORD *)(*(_QWORD *)(v5 + 40) + 8LL),
            *(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(v5 + 40) + 48LL),
            1u);
    if ( v11 )
    {
      if ( ((*(_DWORD *)(v11 + 24) - 214) & 0xFFFFFFFD) == 0 )
      {
        v13 = *(_QWORD *)(v11 + 40);
        v11 = *(_QWORD *)v13;
        v12 = *(_DWORD *)(v13 + 8);
      }
      if ( v4 == v11 && v12 == v8 )
        return 1;
      if ( *(_DWORD *)(v4 + 24) == 186 )
      {
        v18 = *(_QWORD *)(v4 + 40);
        if ( v11 == *(_QWORD *)v18 && v12 == *(_DWORD *)(v18 + 8) )
          return 1;
        if ( v11 == *(_QWORD *)(v18 + 40) && v12 == *(_DWORD *)(v18 + 48) )
          return 1;
      }
    }
    v14 = sub_33DFEB0(
            *(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(v5 + 40) + 48LL),
            **(_QWORD **)(v5 + 40),
            *(_QWORD *)(*(_QWORD *)(v5 + 40) + 8LL),
            1u);
    if ( !v14 )
      return 0;
    if ( ((*(_DWORD *)(v14 + 24) - 214) & 0xFFFFFFFD) == 0 )
    {
      v16 = *(_QWORD *)(v14 + 40);
      v14 = *(_QWORD *)v16;
      v15 = *(_DWORD *)(v16 + 8);
    }
    if ( v4 == v14 && v15 == v8 )
      return 1;
    if ( *(_DWORD *)(v4 + 24) == 186
      && ((v17 = *(_QWORD *)(v4 + 40), v14 == *(_QWORD *)v17) && v15 == *(_DWORD *)(v17 + 8)
       || v14 == *(_QWORD *)(v17 + 40) && v15 == *(_DWORD *)(v17 + 48)) )
    {
      return 1;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
