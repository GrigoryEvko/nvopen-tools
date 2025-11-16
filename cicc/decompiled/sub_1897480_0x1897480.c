// Function: sub_1897480
// Address: 0x1897480
//
__int64 __fastcall sub_1897480(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  _QWORD *v3; // rax
  __int64 v4; // r13
  int v5; // ebx
  int v6; // r12d
  __int64 v7; // rdx
  int v8; // esi
  unsigned int v9; // eax
  __int64 v10; // rcx

  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 )
    return 0;
  while ( 1 )
  {
    v3 = sub_1648700(v2);
    if ( (unsigned __int8)(*((_BYTE *)v3 + 16) - 25) <= 9u )
      break;
    v2 = *(_QWORD *)(v2 + 8);
    if ( !v2 )
      return 0;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_DWORD *)(a1 + 24);
  v6 = v5 - 1;
LABEL_7:
  if ( v5 )
  {
    v7 = v3[5];
    v8 = 1;
    v9 = v6 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v10 = *(_QWORD *)(v4 + 8LL * v9);
    if ( v7 == v10 )
      goto LABEL_5;
    while ( v10 != -8 )
    {
      v9 = v6 & (v8 + v9);
      v10 = *(_QWORD *)(v4 + 8LL * v9);
      if ( v7 == v10 )
      {
LABEL_5:
        while ( 1 )
        {
          v2 = *(_QWORD *)(v2 + 8);
          if ( !v2 )
            return 0;
          v3 = sub_1648700(v2);
          if ( (unsigned __int8)(*((_BYTE *)v3 + 16) - 25) <= 9u )
            goto LABEL_7;
        }
      }
      ++v8;
    }
  }
  return 1;
}
