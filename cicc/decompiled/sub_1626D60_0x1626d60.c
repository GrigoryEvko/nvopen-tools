// Function: sub_1626D60
// Address: 0x1626d60
//
void __fastcall sub_1626D60(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rcx
  int v12; // r11d
  __int64 *v13; // r10
  int v14; // eax
  int v15; // ecx
  __int64 v16; // [rsp-38h] [rbp-38h] BYREF
  __int64 *v17; // [rsp-30h] [rbp-30h] BYREF

  *(_DWORD *)(a2 + 8) = 0;
  if ( (*(_BYTE *)(a1 + 34) & 0x10) != 0 )
  {
    v16 = a1;
    v3 = sub_16498A0(a1);
    v4 = *(_QWORD *)v3;
    v5 = *(_DWORD *)(*(_QWORD *)v3 + 2760LL);
    v6 = *(_QWORD *)v3 + 2736LL;
    if ( v5 )
    {
      v7 = v16;
      v8 = *(_QWORD *)(v4 + 2744);
      v9 = (v5 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v10 = (__int64 *)(v8 + 40 * v9);
      v11 = *v10;
      if ( v16 == *v10 )
      {
LABEL_4:
        sub_1622790(v10 + 1, a2);
        return;
      }
      v12 = 1;
      v13 = 0;
      while ( v11 != -8 )
      {
        if ( v11 == -16 && !v13 )
          v13 = v10;
        LODWORD(v9) = (v5 - 1) & (v12 + v9);
        v10 = (__int64 *)(v8 + 40LL * (unsigned int)v9);
        v11 = *v10;
        if ( v16 == *v10 )
          goto LABEL_4;
        ++v12;
      }
      v14 = *(_DWORD *)(v4 + 2752);
      if ( v13 )
        v10 = v13;
      ++*(_QWORD *)(v4 + 2736);
      v15 = v14 + 1;
      if ( 4 * (v14 + 1) < 3 * v5 )
      {
        if ( v5 - *(_DWORD *)(v4 + 2756) - v15 > v5 >> 3 )
        {
LABEL_11:
          *(_DWORD *)(v4 + 2752) = v15;
          if ( *v10 != -8 )
            --*(_DWORD *)(v4 + 2756);
          *v10 = v7;
          v10[1] = (__int64)(v10 + 3);
          v10[2] = 0x100000000LL;
          goto LABEL_4;
        }
LABEL_16:
        sub_16261B0(v6, v5);
        sub_1621520(v6, &v16, &v17);
        v10 = v17;
        v7 = v16;
        v15 = *(_DWORD *)(v4 + 2752) + 1;
        goto LABEL_11;
      }
    }
    else
    {
      ++*(_QWORD *)(v4 + 2736);
    }
    v5 *= 2;
    goto LABEL_16;
  }
}
