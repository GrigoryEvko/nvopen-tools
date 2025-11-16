// Function: sub_16A5E70
// Address: 0x16a5e70
//
void __fastcall sub_16A5E70(__int64 a1, unsigned int a2)
{
  __int64 v3; // rax
  _QWORD *v4; // r8
  unsigned __int64 v5; // rdx
  char v6; // di
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned int v9; // edx
  size_t v10; // r14
  int v11; // r13d
  __int64 v12; // r15
  unsigned int v13; // eax
  int v14; // esi
  __int64 v15; // r10
  __int64 v16; // r9
  unsigned __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  __int64 v21; // rcx

  if ( a2 )
  {
    v3 = *(unsigned int *)(a1 + 8);
    v4 = *(_QWORD **)a1;
    v5 = *(_QWORD *)a1;
    v6 = (v3 - 1) & 0x3F;
    if ( (unsigned int)v3 > 0x40 )
      v5 = v4[(unsigned int)(v3 - 1) >> 6];
    v7 = v5 & (1LL << ((unsigned __int8)v3 - 1));
    v8 = (unsigned __int64)(v3 + 63) >> 6;
    v9 = a2 >> 6;
    v10 = 8 * (a2 >> 6);
    v11 = v8 - (a2 >> 6);
    if ( (_DWORD)v8 != a2 >> 6 )
    {
      v13 = v8 - 1;
      v12 = v13;
      v4[v13] = (__int64)(v4[v13] << (63 - v6)) >> (63 - v6);
      v14 = a2 & 0x3F;
      if ( v14 )
      {
        if ( v11 == 1 )
        {
          v18 = 0;
        }
        else
        {
          v15 = 0;
          v16 = v9;
          do
          {
            v17 = *(_QWORD *)(*(_QWORD *)a1 + 8 * v16);
            v16 = ++v9;
            *(_QWORD *)(*(_QWORD *)a1 + v15) = (v17 >> v14)
                                             | (*(_QWORD *)(*(_QWORD *)a1 + 8LL * v9) << (64 - (unsigned __int8)v14));
            v15 += 8;
          }
          while ( v13 != v9 );
          v18 = 8LL * (unsigned int)(v11 - 1);
        }
        *(_QWORD *)(*(_QWORD *)a1 + v18) = *(_QWORD *)(*(_QWORD *)a1 + 8 * v12) >> v14;
        *(_QWORD *)(*(_QWORD *)a1 + v18) = (__int64)(*(_QWORD *)(*(_QWORD *)a1 + v18) << v14) >> v14;
        v4 = *(_QWORD **)a1;
      }
      else
      {
        memmove(*(void **)a1, (const void *)(*(_QWORD *)a1 + v10), (unsigned int)(8 * v11));
        v4 = *(_QWORD **)a1;
      }
    }
    memset(&v4[v11], -(v7 != 0), v10);
    v19 = *(unsigned int *)(a1 + 8);
    v20 = 0xFFFFFFFFFFFFFFFFLL >> -*(_BYTE *)(a1 + 8);
    if ( (unsigned int)v19 <= 0x40 )
    {
      *(_QWORD *)a1 &= v20;
    }
    else
    {
      v21 = (unsigned int)((unsigned __int64)(v19 + 63) >> 6) - 1;
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v21) &= v20;
    }
  }
}
