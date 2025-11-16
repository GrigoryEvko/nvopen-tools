// Function: sub_1DB3D00
// Address: 0x1db3d00
//
__int64 __fastcall sub_1DB3D00(__int64 **a1, __int64 a2, __int64 *a3)
{
  __int64 *v3; // r11
  __int64 *v4; // r10
  __int64 *v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rax
  unsigned int v9; // edx
  unsigned int v10; // eax
  __int64 v11; // rcx
  int v12; // esi
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 *v17; // rax

  v3 = a3;
  v4 = *a1;
  v5 = *(__int64 **)a2;
  v6 = (__int64)&(*a1)[3 * *((unsigned int *)a1 + 2)];
  v7 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  v8 = *a3;
  v9 = *(_DWORD *)((**a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (**a1 >> 1) & 3;
  v10 = *(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v8 >> 1) & 3;
  if ( v9 < v10 )
  {
    v17 = sub_1DB3750(*a1, v6, v3);
    if ( v4 != v17 )
      v4 = v17 - 3;
  }
  else
  {
    if ( v9 <= v10 )
      return 1;
    if ( (__int64 *)v7 != v3 + 3
      && v9 >= (*(_DWORD *)((v3[3] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v3[3] >> 1) & 3) )
    {
      v3 = sub_1DB3750(v3, *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8), v4);
      if ( v5 != v3 )
        v3 -= 3;
    }
  }
  if ( (__int64 *)v7 != v3 && (__int64 *)v6 != v4 )
  {
    v11 = *v3;
    v12 = *(_DWORD *)((*v3 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    while ( 1 )
    {
      v13 = *(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v4 >> 1) & 3;
      if ( v13 <= (v12 | (unsigned int)(v11 >> 1) & 3) )
      {
        v13 = v12 | (v11 >> 1) & 3;
      }
      else
      {
        v14 = v6;
        v6 = v7;
        v12 = *(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24);
        v11 = *v4;
        v7 = v14;
        v15 = v4;
        v4 = v3;
        v3 = v15;
      }
      if ( (*(_DWORD *)((v4[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v4[1] >> 1) & 3) > v13 )
        break;
      v4 += 3;
      if ( v4 == (__int64 *)v6 )
        return 0;
    }
    return 1;
  }
  return 0;
}
