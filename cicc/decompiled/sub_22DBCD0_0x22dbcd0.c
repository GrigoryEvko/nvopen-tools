// Function: sub_22DBCD0
// Address: 0x22dbcd0
//
__int64 __fastcall sub_22DBCD0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r9
  __int64 v5; // r8
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r11
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned int v12; // eax
  int v14; // edx
  int v15; // ebx
  __int64 *v16; // rdx
  __int64 v17; // r11
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned int v21; // eax

  v3 = *(unsigned int *)(a3 + 24);
  v4 = *a2;
  v5 = *(_QWORD *)(a3 + 8);
  if ( !(_DWORD)v3 )
    return a2[1];
  v6 = (v3 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v7 = (__int64 *)(v5 + 16LL * v6);
  v8 = *v7;
  if ( v4 == *v7 )
  {
    if ( v7 != (__int64 *)(v5 + 16 * v3) )
    {
      v9 = v7[1];
      v10 = *(_QWORD *)(a1 + 16);
      if ( v9 )
      {
        v11 = (unsigned int)(*(_DWORD *)(v9 + 44) + 1);
        v12 = *(_DWORD *)(v9 + 44) + 1;
      }
      else
      {
        v11 = 0;
        v12 = 0;
      }
      if ( v12 >= *(_DWORD *)(v10 + 56) )
        BUG();
      return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v10 + 48) + 8 * v11) + 8LL);
    }
    return a2[1];
  }
  v14 = 1;
  if ( v8 == -4096 )
    return a2[1];
  while ( 1 )
  {
    v15 = v14 + 1;
    v6 = (v3 - 1) & (v14 + v6);
    v16 = (__int64 *)(v5 + 16LL * v6);
    v17 = *v16;
    if ( v4 == *v16 )
      break;
    v14 = v15;
    if ( v17 == -4096 )
      return a2[1];
  }
  if ( v16 == (__int64 *)(v5 + 16 * v3) )
    return a2[1];
  v18 = v16[1];
  v19 = *(_QWORD *)(a1 + 16);
  if ( v18 )
  {
    v20 = (unsigned int)(*(_DWORD *)(v18 + 44) + 1);
    v21 = *(_DWORD *)(v18 + 44) + 1;
  }
  else
  {
    v20 = 0;
    v21 = 0;
  }
  if ( v21 >= *(_DWORD *)(v19 + 56) )
    BUG();
  return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v19 + 48) + 8 * v20) + 8LL);
}
