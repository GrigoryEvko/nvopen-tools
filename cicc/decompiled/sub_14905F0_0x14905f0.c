// Function: sub_14905F0
// Address: 0x14905f0
//
void __fastcall sub_14905F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6)
{
  int v6; // r12d
  int v9; // r12d
  __int64 *v10; // r8
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 *v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // [rsp+0h] [rbp-70h]
  int v17; // [rsp+14h] [rbp-5Ch]
  __int64 *v18; // [rsp+18h] [rbp-58h]
  __int64 v19; // [rsp+28h] [rbp-48h] BYREF
  __int64 v20; // [rsp+30h] [rbp-40h] BYREF
  __int64 v21; // [rsp+38h] [rbp-38h] BYREF

  v6 = *(_DWORD *)(a4 + 8);
  if ( v6 && (*(_WORD *)(a2 + 24) != 7 || *(_QWORD *)(a2 + 40) == 2) )
  {
    v9 = v6 - 1;
    v19 = a2;
    v17 = v9;
    if ( v9 >= 0 )
    {
      v10 = &v21;
      v11 = 8LL * v9;
      v16 = a3 + 16;
      while ( 1 )
      {
        v18 = v10;
        sub_148F0C0(a1, a2, *(_QWORD *)(*(_QWORD *)a4 + v11), &v20, v10, a5, a6);
        v10 = v18;
        v19 = v20;
        if ( v17 == v9 )
        {
          if ( *(_WORD *)(v21 + 24) == 7 )
          {
            *(_DWORD *)(a3 + 8) = 0;
            *(_DWORD *)(a4 + 8) = 0;
            return;
          }
        }
        else
        {
          v12 = *(unsigned int *)(a3 + 8);
          if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 12) )
          {
            sub_16CD150(a3, v16, 0, 8);
            v12 = *(unsigned int *)(a3 + 8);
            v10 = v18;
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v21;
          ++*(_DWORD *)(a3 + 8);
        }
        --v9;
        v11 -= 8;
        if ( v9 == -1 )
          break;
        a2 = v19;
      }
    }
    sub_1458920(a3, &v19);
    v13 = *(__int64 **)a3;
    v14 = (__int64 *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8));
    if ( *(__int64 **)a3 != v14 )
    {
      while ( v13 < --v14 )
      {
        v15 = *v13++;
        *(v13 - 1) = *v14;
        *v14 = v15;
      }
    }
  }
}
