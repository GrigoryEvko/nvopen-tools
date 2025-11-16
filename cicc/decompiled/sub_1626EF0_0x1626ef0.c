// Function: sub_1626EF0
// Address: 0x1626ef0
//
__int64 __fastcall sub_1626EF0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // r13
  unsigned int v7; // esi
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rax
  char **v12; // rbx
  char *v13; // rcx
  int v14; // r10d
  char **v15; // r9
  int v16; // eax
  int v17; // ecx
  unsigned __int8 v18; // [rsp-49h] [rbp-49h]
  char *v19; // [rsp-48h] [rbp-48h] BYREF
  char **v20; // [rsp-40h] [rbp-40h] BYREF

  result = 0;
  if ( (*(_BYTE *)(a1 + 34) & 0x10) != 0 )
  {
    v19 = (char *)a1;
    v5 = sub_16498A0(a1);
    v6 = *(_QWORD *)v5;
    v7 = *(_DWORD *)(*(_QWORD *)v5 + 2760LL);
    v8 = *(_QWORD *)v5 + 2736LL;
    if ( v7 )
    {
      v9 = (__int64)v19;
      v10 = *(_QWORD *)(v6 + 2744);
      v11 = (v7 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v12 = (char **)(v10 + 40 * v11);
      v13 = *v12;
      if ( v19 == *v12 )
      {
LABEL_4:
        result = sub_16236C0(v12 + 1, a2);
        if ( !*((_DWORD *)v12 + 4) )
        {
          v18 = result;
          sub_161FB70(a1);
          return v18;
        }
        return result;
      }
      v14 = 1;
      v15 = 0;
      while ( v13 != (char *)-8LL )
      {
        if ( !v15 && v13 == (char *)-16LL )
          v15 = v12;
        LODWORD(v11) = (v7 - 1) & (v14 + v11);
        v12 = (char **)(v10 + 40LL * (unsigned int)v11);
        v13 = *v12;
        if ( v19 == *v12 )
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
LABEL_12:
          *(_DWORD *)(v6 + 2752) = v17;
          if ( *v12 != (char *)-8LL )
            --*(_DWORD *)(v6 + 2756);
          *v12 = (char *)v9;
          v12[1] = (char *)(v12 + 3);
          v12[2] = (char *)0x100000000LL;
          goto LABEL_4;
        }
LABEL_17:
        sub_16261B0(v8, v7);
        sub_1621520(v8, (__int64 *)&v19, &v20);
        v12 = v20;
        v9 = (__int64)v19;
        v17 = *(_DWORD *)(v6 + 2752) + 1;
        goto LABEL_12;
      }
    }
    else
    {
      ++*(_QWORD *)(v6 + 2736);
    }
    v7 *= 2;
    goto LABEL_17;
  }
  return result;
}
