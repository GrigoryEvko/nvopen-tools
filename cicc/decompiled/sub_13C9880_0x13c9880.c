// Function: sub_13C9880
// Address: 0x13c9880
//
__int64 __fastcall sub_13C9880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r14
  __int16 v9; // ax
  _QWORD *v11; // rbx
  int v12; // eax
  _QWORD *v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  _QWORD *v16; // rbx
  __int64 v17; // rax
  int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rcx
  int v21; // edi
  __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  char v29; // al
  char v30; // al
  _QWORD *v31; // rdx
  int v32; // eax
  int v33; // r9d
  _QWORD *v34; // [rsp+0h] [rbp-40h]
  __int64 v35; // [rsp+0h] [rbp-40h]
  __int64 v36; // [rsp+8h] [rbp-38h]
  __int64 v37; // [rsp+8h] [rbp-38h]
  __int64 v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]
  __int64 v40; // [rsp+8h] [rbp-38h]

  v6 = a1;
  v9 = *(_WORD *)(a1 + 24);
  if ( v9 == 7 )
  {
LABEL_2:
    if ( a3 != *(_QWORD *)(v6 + 48) )
    {
      v36 = a3;
      if ( (unsigned __int8)sub_13C9880(**(_QWORD **)(v6 + 32), a2, a3, a4, a5) )
      {
        v26 = sub_13A5BC0((_QWORD *)v6, a4);
        LODWORD(v6) = sub_13C9880(v26, a2, v36, a4, a5) ^ 1;
        return (unsigned int)v6;
      }
      goto LABEL_4;
    }
    if ( *(_QWORD *)(v6 + 40) == 2 )
    {
LABEL_7:
      LODWORD(v6) = 1;
      return (unsigned int)v6;
    }
    v13 = *(_QWORD **)(a3 + 72);
    v14 = *(_QWORD **)(a3 + 64);
    v15 = *(_QWORD *)(a2 + 40);
    if ( v13 == v14 )
    {
      v16 = &v14[*(unsigned int *)(a3 + 84)];
      if ( v14 == v16 )
      {
        v31 = *(_QWORD **)(a3 + 64);
      }
      else
      {
        do
        {
          if ( v15 == *v14 )
            break;
          ++v14;
        }
        while ( v16 != v14 );
        v31 = v16;
      }
    }
    else
    {
      v35 = a3;
      v38 = *(_QWORD *)(a2 + 40);
      v16 = &v13[*(unsigned int *)(a3 + 80)];
      v14 = (_QWORD *)sub_16CC9F0(a3 + 56, v15);
      if ( v38 == *v14 )
      {
        v27 = *(_QWORD *)(v35 + 72);
        if ( v27 == *(_QWORD *)(v35 + 64) )
          v31 = (_QWORD *)(v27 + 8LL * *(unsigned int *)(v35 + 84));
        else
          v31 = (_QWORD *)(v27 + 8LL * *(unsigned int *)(v35 + 80));
      }
      else
      {
        v17 = *(_QWORD *)(v35 + 72);
        if ( v17 != *(_QWORD *)(v35 + 64) )
        {
          v14 = (_QWORD *)(v17 + 8LL * *(unsigned int *)(v35 + 80));
          goto LABEL_26;
        }
        v14 = (_QWORD *)(v17 + 8LL * *(unsigned int *)(v35 + 84));
        v31 = v14;
      }
    }
    while ( v31 != v14 && *v14 >= 0xFFFFFFFFFFFFFFFELL )
      ++v14;
LABEL_26:
    if ( v16 == v14 )
    {
      v18 = *(_DWORD *)(a5 + 24);
      v19 = 0;
      if ( v18 )
      {
        v20 = *(_QWORD *)(a2 + 40);
        v21 = v18 - 1;
        v22 = *(_QWORD *)(a5 + 8);
        v23 = (v18 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v24 = (__int64 *)(v22 + 16LL * v23);
        v25 = *v24;
        if ( v20 == *v24 )
        {
LABEL_29:
          v19 = v24[1];
        }
        else
        {
          v32 = 1;
          while ( v25 != -8 )
          {
            v33 = v32 + 1;
            v23 = v21 & (v32 + v23);
            v24 = (__int64 *)(v22 + 16LL * v23);
            v25 = *v24;
            if ( v20 == *v24 )
              goto LABEL_29;
            v32 = v33;
          }
          v19 = 0;
        }
      }
      LOBYTE(v6) = v6 != sub_1472270(a4, v6, v19);
      return (unsigned int)v6;
    }
LABEL_4:
    LODWORD(v6) = 0;
    return (unsigned int)v6;
  }
  while ( v9 != 4 )
  {
    if ( v9 == 5 && *(_QWORD *)(v6 + 40) == 2 )
    {
      v40 = a3;
      v29 = sub_146CEE0(a4, **(_QWORD **)(v6 + 32), a3);
      a3 = v40;
      if ( v29 )
      {
        v30 = sub_13C9880(*(_QWORD *)(*(_QWORD *)(v6 + 32) + 8LL), a2, v40, a4, a5);
        a3 = v40;
        if ( v30 )
          goto LABEL_7;
      }
    }
    if ( byte_4F99000 )
      goto LABEL_4;
    if ( *(_WORD *)(v6 + 24) != 3 )
      goto LABEL_4;
    if ( byte_4F990E0 )
    {
      v39 = a3;
      v28 = sub_1BF8840(v6, a4, a3, 0, 1);
      a3 = v39;
      if ( v6 == v28 )
        goto LABEL_4;
    }
    v6 = *(_QWORD *)(v6 + 32);
    v9 = *(_WORD *)(v6 + 24);
    if ( v9 == 7 )
      goto LABEL_2;
  }
  v11 = *(_QWORD **)(v6 + 32);
  v34 = &v11[*(_QWORD *)(v6 + 40)];
  if ( v11 == v34 )
    goto LABEL_4;
  LODWORD(v6) = 0;
  do
  {
    v37 = a3;
    v12 = sub_13C9880(*v11, a2, a3, a4, a5);
    a3 = v37;
    if ( (_BYTE)v12 )
    {
      if ( (_BYTE)v6 )
        goto LABEL_4;
      LODWORD(v6) = v12;
    }
    ++v11;
  }
  while ( v34 != v11 );
  return (unsigned int)v6;
}
