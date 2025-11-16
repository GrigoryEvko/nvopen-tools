// Function: sub_1626560
// Address: 0x1626560
//
void __fastcall sub_1626560(__int64 a1, int a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned int v7; // esi
  __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // rcx
  int v14; // r11d
  __int64 *v15; // r10
  int v16; // eax
  int v17; // ecx
  __int64 v18; // [rsp-48h] [rbp-48h] BYREF
  __int64 *v19; // [rsp-40h] [rbp-40h] BYREF

  if ( (*(_BYTE *)(a1 + 34) & 0x10) != 0 )
  {
    v18 = a1;
    v5 = sub_16498A0(a1);
    v6 = *(_QWORD *)v5;
    v7 = *(_DWORD *)(*(_QWORD *)v5 + 2760LL);
    v8 = *(_QWORD *)v5 + 2736LL;
    if ( v7 )
    {
      v9 = v18;
      v10 = *(_QWORD *)(v6 + 2744);
      v11 = (v7 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v12 = (__int64 *)(v10 + 40 * v11);
      v13 = *v12;
      if ( v18 == *v12 )
      {
LABEL_4:
        sub_161F7A0(v12 + 1, a2, a3);
        return;
      }
      v14 = 1;
      v15 = 0;
      while ( v13 != -8 )
      {
        if ( v13 == -16 && !v15 )
          v15 = v12;
        LODWORD(v11) = (v7 - 1) & (v14 + v11);
        v12 = (__int64 *)(v10 + 40LL * (unsigned int)v11);
        v13 = *v12;
        if ( v18 == *v12 )
          goto LABEL_4;
        ++v14;
      }
      v16 = *(_DWORD *)(v6 + 2752);
      if ( v15 )
        v12 = v15;
      ++*(_QWORD *)(v6 + 2736);
      v17 = v16 + 1;
      if ( 4 * (v16 + 1) < 3 * v7 )
      {
        if ( v7 - *(_DWORD *)(v6 + 2756) - v17 > v7 >> 3 )
        {
LABEL_11:
          *(_DWORD *)(v6 + 2752) = v17;
          if ( *v12 != -8 )
            --*(_DWORD *)(v6 + 2756);
          *v12 = v9;
          v12[1] = (__int64)(v12 + 3);
          v12[2] = 0x100000000LL;
          goto LABEL_4;
        }
LABEL_16:
        sub_16261B0(v8, v7);
        sub_1621520(v8, &v18, &v19);
        v12 = v19;
        v9 = v18;
        v17 = *(_DWORD *)(v6 + 2752) + 1;
        goto LABEL_11;
      }
    }
    else
    {
      ++*(_QWORD *)(v6 + 2736);
    }
    v7 *= 2;
    goto LABEL_16;
  }
}
