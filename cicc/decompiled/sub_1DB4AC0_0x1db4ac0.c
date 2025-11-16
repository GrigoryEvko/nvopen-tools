// Function: sub_1DB4AC0
// Address: 0x1db4ac0
//
__int64 __fastcall sub_1DB4AC0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r12
  __int64 *v4; // rbx
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdi
  unsigned int v9; // esi
  unsigned int v10; // ecx
  __int64 v11; // rax

  v3 = &a2[a3];
  if ( v3 != a2 )
  {
    v4 = a2;
    v6 = (__int64 *)sub_1DB3C70((__int64 *)a1, *a2);
    v7 = 24LL * *(unsigned int *)(a1 + 8);
    v8 = (__int64 *)(*(_QWORD *)a1 + v7);
    if ( v8 != v6 )
    {
      v9 = *(_DWORD *)((*(_QWORD *)(*(_QWORD *)a1 + v7 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
         | (*(__int64 *)(*(_QWORD *)a1 + v7 - 16) >> 1) & 3;
      do
      {
        v10 = *(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v4 >> 1) & 3;
        if ( v10 >= v9 )
          break;
        if ( v10 >= (*(_DWORD *)((v6[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v6[1] >> 1) & 3) )
        {
          do
          {
            v11 = v6[4];
            v6 += 3;
          }
          while ( v10 >= (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11 >> 1) & 3) );
        }
        if ( v8 == v6 )
          break;
        if ( v10 >= (*(_DWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v6 >> 1) & 3)
          && v10 < (*(_DWORD *)((v6[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v6[1] >> 1) & 3) )
        {
          return 1;
        }
        ++v4;
      }
      while ( v3 != v4 );
    }
  }
  return 0;
}
