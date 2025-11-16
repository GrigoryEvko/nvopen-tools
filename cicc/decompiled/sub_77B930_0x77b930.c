// Function: sub_77B930
// Address: 0x77b930
//
__int64 __fastcall sub_77B930(__int64 a1, __int64 a2, __int64 a3, _WORD **a4)
{
  __int64 v4; // r10
  __int64 v5; // rax
  unsigned int v6; // r8d
  _QWORD *v7; // r14
  __int64 v8; // r12
  int v10; // ebx
  char *v11; // rax
  const char *v12; // r15
  char *v13; // rax
  __int64 v14; // r13
  const char *v15; // r12
  char *v16; // rax
  _WORD *v17; // rbx
  __int64 v18; // r15
  char j; // al
  unsigned int v20; // ecx
  unsigned int v21; // edx
  unsigned int v22; // eax
  int v23; // eax
  unsigned int v24; // ecx
  unsigned int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // [rsp+0h] [rbp-60h]
  unsigned int i; // [rsp+14h] [rbp-4Ch] BYREF
  unsigned int v30; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v31; // [rsp+1Ch] [rbp-44h] BYREF
  __int64 v32; // [rsp+20h] [rbp-40h] BYREF
  __int64 v33[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = a1;
  v5 = *(_QWORD *)(a2 + 152);
  for ( i = 1; *(_BYTE *)(v5 + 140) == 12; v5 = *(_QWORD *)(v5 + 160) )
    ;
  v6 = 0;
  if ( (*(_BYTE *)(a1 + 132) & 1) != 0 )
  {
    v7 = **(_QWORD ***)(v5 + 168);
    if ( (*(_BYTE *)(a1 + 133) & 0x20) == 0 )
    {
      sub_729E00(*(_DWORD *)(a1 + 112), v33, &v32, &v30, &v31);
      v11 = sub_67C860(2997);
      fprintf(qword_4F07510, "\n%s\n", v11);
      v4 = a1;
      if ( v30 )
      {
        v12 = (const char *)v33[0];
        v13 = sub_67C860(1459);
        v14 = v30;
        v15 = v13;
        v16 = sub_67C860(1458);
        fprintf(qword_4F07510, "%s%lu%s%s\n", v16, v14, v15, v12);
        v4 = a1;
      }
      *(_BYTE *)(v4 + 133) |= 0x20u;
    }
    v27 = v4;
    if ( (unsigned int)sub_8D2780(v7[1]) )
    {
      v10 = sub_8D27E0(v7[1]);
      sub_620E00(*a4, v10, v33, (int *)&v32);
      if ( (_DWORD)v32 )
      {
        fwrite("(overflow)", 1u, 0xAu, qword_4F07510);
      }
      else if ( v10 )
      {
        fprintf(qword_4F07510, "%lld", v33[0]);
      }
      else
      {
        fprintf(qword_4F07510, "%llu", v33[0]);
      }
      return i;
    }
    v8 = (__int64)*a4;
    if ( ((*a4)[4] & 0xA) != 8 )
    {
      fwrite("(invalid string pointer)", 1u, 0x18u, qword_4F07510);
      return i;
    }
    v17 = *(_WORD **)v8;
    v18 = sub_8D46C0(v7[1]);
    for ( j = *(_BYTE *)(v18 + 140); j == 12; j = *(_BYTE *)(v18 + 140) )
      v18 = *(_QWORD *)(v18 + 160);
    if ( (*(_BYTE *)(v8 + 8) & 1) != 0 )
    {
      v20 = 1;
      if ( j != 1 )
        v20 = *(_DWORD *)(v18 + 128);
      sub_771560(v27, *(_QWORD *)(v8 + 16), v18, v20, &v31, &v30, &i);
    }
    else
    {
      v24 = 16;
      if ( (unsigned __int8)(j - 2) > 1u )
        v24 = sub_7764B0(v27, v18, &i);
      v6 = i;
      if ( !i )
      {
        v31 = 0;
        v30 = 0;
        if ( !*v7 )
          return v6;
        goto LABEL_34;
      }
      v25 = *(unsigned __int8 *)(v8 + 8);
      if ( (v25 & 8) != 0 )
      {
        v31 = *(_DWORD *)(v8 + 8) >> 8;
        v26 = *(_QWORD *)(v8 + 16);
        if ( (v25 & 4) != 0 )
          v26 = *(_QWORD *)(v26 + 24);
        if ( v24 )
          v30 = ((unsigned int)*(_QWORD *)v8 - (unsigned int)v26) / v24;
        else
          v30 = 0;
      }
      else
      {
        v31 = 1;
        v30 = (v25 >> 1) & 1;
      }
    }
    if ( !*v7 )
    {
      v21 = v31;
      v22 = v30;
      goto LABEL_25;
    }
LABEL_34:
    sub_620E00(a4[1], 1, v33, (int *)&v32);
    if ( (_DWORD)v32 )
    {
      v22 = v30;
      v21 = v31;
    }
    else
    {
      v22 = v30;
      if ( v33[0] < 0 )
        return i;
      v21 = v31;
      if ( LODWORD(v33[0]) < v31 - v30 )
      {
        v21 = LODWORD(v33[0]) - v30;
        v31 = LODWORD(v33[0]) - v30;
      }
    }
LABEL_25:
    if ( v21 > v22 )
    {
      do
      {
        v23 = sub_8D27E0(v18);
        sub_620E00(v17, v23, v33, (int *)&v32);
        if ( !v33[0] )
          break;
        v17 += 8;
        fputc(SLOBYTE(v33[0]), qword_4F07510);
        ++v30;
      }
      while ( v31 > v30 );
    }
    return i;
  }
  return v6;
}
