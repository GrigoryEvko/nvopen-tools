// Function: sub_2E0ADA0
// Address: 0x2e0ada0
//
__int64 __fastcall sub_2E0ADA0(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r12
  __int64 *v4; // rbx
  __int64 *v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rdi
  __int64 v9; // r8
  __int64 v10; // rsi
  unsigned __int64 v11; // r8
  unsigned int v12; // esi
  unsigned int v13; // ecx
  __int64 v14; // rax

  v3 = &a2[a3];
  if ( v3 != a2 )
  {
    v4 = a2;
    v6 = (__int64 *)sub_2E09D00((__int64 *)a1, *a2);
    v7 = 24LL * *(unsigned int *)(a1 + 8);
    v8 = (__int64 *)(*(_QWORD *)a1 + v7);
    if ( v8 != v6 )
    {
      v9 = *(_QWORD *)(*(_QWORD *)a1 + v7 - 16);
      v10 = v9 >> 1;
      v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      v12 = v10 & 3;
      do
      {
        v13 = *(_DWORD *)((*v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v4 >> 1) & 3;
        if ( v13 >= (v12 | *(_DWORD *)(v11 + 24)) )
          break;
        if ( (*(_DWORD *)((v6[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v6[1] >> 1) & 3) <= v13 )
        {
          do
          {
            v14 = v6[4];
            v6 += 3;
          }
          while ( v13 >= (*(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v14 >> 1) & 3) );
        }
        if ( v8 == v6 )
          break;
        if ( v13 >= (*(_DWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v6 >> 1) & 3)
          && v13 < (*(_DWORD *)((v6[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v6[1] >> 1) & 3) )
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
