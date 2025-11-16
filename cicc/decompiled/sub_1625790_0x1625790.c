// Function: sub_1625790
// Address: 0x1625790
//
__int64 __fastcall sub_1625790(__int64 a1, int a2)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  unsigned int v6; // esi
  __int64 v7; // r13
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // rdx
  int v13; // r11d
  __int64 *v14; // r10
  int v15; // eax
  int v16; // edx
  __int64 v17; // [rsp-38h] [rbp-38h] BYREF
  __int64 *v18; // [rsp-30h] [rbp-30h] BYREF

  if ( !a2 )
    return *(_QWORD *)(a1 + 48);
  if ( *(__int16 *)(a1 + 18) < 0 )
  {
    v17 = a1;
    v4 = sub_16498A0(a1);
    v5 = *(_QWORD *)v4;
    v6 = *(_DWORD *)(*(_QWORD *)v4 + 2728LL);
    v7 = *(_QWORD *)v4 + 2704LL;
    if ( v6 )
    {
      v8 = v17;
      v9 = *(_QWORD *)(v5 + 2712);
      v10 = (v6 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v11 = (__int64 *)(v9 + 56 * v10);
      v12 = *v11;
      if ( v17 == *v11 )
        return sub_161F650(v11 + 1, a2);
      v13 = 1;
      v14 = 0;
      while ( v12 != -8 )
      {
        if ( v12 == -16 && !v14 )
          v14 = v11;
        LODWORD(v10) = (v6 - 1) & (v13 + v10);
        v11 = (__int64 *)(v9 + 56LL * (unsigned int)v10);
        v12 = *v11;
        if ( v17 == *v11 )
          return sub_161F650(v11 + 1, a2);
        ++v13;
      }
      v15 = *(_DWORD *)(v5 + 2720);
      if ( v14 )
        v11 = v14;
      ++*(_QWORD *)(v5 + 2704);
      v16 = v15 + 1;
      if ( 4 * (v15 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(v5 + 2724) - v16 > v6 >> 3 )
        {
LABEL_14:
          *(_DWORD *)(v5 + 2720) = v16;
          if ( *v11 != -8 )
            --*(_DWORD *)(v5 + 2724);
          *v11 = v8;
          v11[1] = (__int64)(v11 + 3);
          v11[2] = 0x200000000LL;
          return sub_161F650(v11 + 1, a2);
        }
LABEL_19:
        sub_1624590(v7, v6);
        sub_1621460(v7, &v17, &v18);
        v11 = v18;
        v8 = v17;
        v16 = *(_DWORD *)(v5 + 2720) + 1;
        goto LABEL_14;
      }
    }
    else
    {
      ++*(_QWORD *)(v5 + 2704);
    }
    v6 *= 2;
    goto LABEL_19;
  }
  return 0;
}
