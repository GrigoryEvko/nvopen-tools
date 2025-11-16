// Function: sub_2F52670
// Address: 0x2f52670
//
__int64 __fastcall sub_2F52670(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // r13
  __int64 v3; // r12
  _QWORD *v4; // rbx
  __int64 v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rcx
  int v9; // ebx
  int v10; // esi
  __int64 v11; // rax
  bool v12; // cf
  __int64 v13; // r9
  __int64 v14; // r15
  unsigned int *v15; // r14
  __int64 v16; // rax
  unsigned int v17; // edi
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r8
  unsigned int v21; // ecx
  unsigned int v22; // r8d
  __int64 v23; // r10
  __int64 v24; // r9
  int *v25; // rsi
  unsigned int *v26; // r14
  unsigned int *i; // r8
  __int64 v28; // rax
  __int64 v29; // r15
  __int64 v30; // rdi
  __int64 v31; // r10
  unsigned int v32; // ecx
  __int128 v33; // rax
  __int64 v34; // rdx
  int *v35; // rax
  __int64 v36; // rax
  __int64 v37; // r12
  _QWORD *v39; // [rsp+0h] [rbp-70h]
  __int64 v40; // [rsp+8h] [rbp-68h]
  __int64 v41; // [rsp+10h] [rbp-60h]
  char v42; // [rsp+18h] [rbp-58h]
  char v43; // [rsp+1Ch] [rbp-54h]
  __int64 v44; // [rsp+20h] [rbp-50h]
  unsigned int *v45; // [rsp+28h] [rbp-48h]
  __int64 v46; // [rsp+30h] [rbp-40h]
  __int64 v47; // [rsp+30h] [rbp-40h]
  __int64 v48; // [rsp+38h] [rbp-38h]

  v2 = a1;
  v3 = 0;
  v4 = a2;
  v5 = a1[124];
  v46 = *(_QWORD *)(v5 + 280);
  v48 = *(unsigned int *)(v5 + 288);
  if ( *(_DWORD *)(v5 + 288) )
  {
    v6 = a1;
    v7 = 0;
    v9 = 0;
    while ( 1 )
    {
      v13 = a2[3];
      v14 = a2[1];
      v15 = (unsigned int *)(v6[3012] + 8 * v7);
      v16 = v46 + 40 * v7;
      v17 = *v15;
      v18 = *(_QWORD *)(v6[103] + 8LL);
      v19 = 2 * *v15;
      v20 = (unsigned int)(v19 + 1);
      v21 = *(_DWORD *)(v18 + 4 * v19);
      v22 = *(_DWORD *)(v18 + 4 * v20);
      v23 = *(_QWORD *)(v13 + 8LL * (v21 >> 6));
      v24 = *(_QWORD *)(v13 + 8LL * (v22 >> 6));
      v25 = &dword_503BD90;
      if ( v14 )
      {
        v25 = (int *)(*(_QWORD *)(v14 + 512) + 24LL * v17);
        if ( *v25 != *(_DWORD *)(v14 + 4) )
        {
          v39 = v6;
          v40 = v16;
          v42 = v21;
          v41 = v23;
          v43 = v22;
          v44 = v24;
          sub_3501C20(v14);
          v6 = v39;
          v16 = v40;
          LOBYTE(v21) = v42;
          v23 = v41;
          v25 = (int *)(24LL * v17 + *(_QWORD *)(v14 + 512));
          LOBYTE(v22) = v43;
          v24 = v44;
        }
      }
      a2[2] = v25;
      if ( *(_BYTE *)(v16 + 32) )
        break;
      if ( *(_BYTE *)(v16 + 33) )
      {
        v10 = 0;
        goto LABEL_5;
      }
LABEL_12:
      v7 = (unsigned int)++v9;
      if ( v9 == v48 )
      {
        v4 = a2;
        v2 = v6;
        goto LABEL_20;
      }
    }
    v10 = (*((_BYTE *)v15 + 4) == 1) ^ ((v23 & (1LL << v21)) != 0);
    if ( *(_BYTE *)(v16 + 33) )
LABEL_5:
      v10 += (*((_BYTE *)v15 + 5) == 1) ^ ((v24 & (1LL << v22)) != 0);
    if ( v10 )
    {
      v11 = *(_QWORD *)(*(_QWORD *)(v6[104] + 136LL) + 8LL * *v15);
      v12 = __CFADD__(v11, v3);
      v3 += v11;
      if ( v12 )
        v3 = -1;
      if ( v10 != 1 )
      {
        v12 = __CFADD__(v11, v3);
        v3 += v11;
        if ( v12 )
          v3 = -1;
      }
    }
    goto LABEL_12;
  }
LABEL_20:
  v26 = (unsigned int *)v4[12];
  for ( i = &v26[*((unsigned int *)v4 + 26)]; i != v26; ++v26 )
  {
    while ( 1 )
    {
      v29 = *v26;
      v30 = v4[3];
      v31 = *(_QWORD *)(v2[103] + 8LL);
      v32 = *(_DWORD *)(v31 + 4LL * (unsigned int)(2 * v29 + 1));
      *((_QWORD *)&v33 + 1) = *(_QWORD *)(v30 + 8LL * (*(_DWORD *)(v31 + 4LL * (unsigned int)(2 * v29)) >> 6))
                            & (1LL << *(_DWORD *)(v31 + 4LL * (unsigned int)(2 * v29)));
      *(_QWORD *)&v33 = *(_QWORD *)(v30 + 8LL * (v32 >> 6)) & (1LL << v32);
      if ( v33 == 0 )
        goto LABEL_24;
      if ( (_QWORD)v33 && *((_QWORD *)&v33 + 1) )
        break;
      v28 = *(_QWORD *)(v2[104] + 136LL);
      v12 = __CFADD__(*(_QWORD *)(v28 + 8 * v29), v3);
      v3 += *(_QWORD *)(v28 + 8 * v29);
      if ( v12 )
        goto LABEL_23;
LABEL_24:
      if ( i == ++v26 )
        return v3;
    }
    v34 = v4[1];
    v35 = &dword_503BD90;
    if ( v34 )
    {
      v35 = (int *)(*(_QWORD *)(v34 + 512) + 24LL * (unsigned int)v29);
      if ( *v35 != *(_DWORD *)(v34 + 4) )
      {
        v45 = i;
        v47 = v4[1];
        sub_3501C20(v34);
        i = v45;
        v35 = (int *)(*(_QWORD *)(v47 + 512) + 24LL * (unsigned int)v29);
      }
    }
    v4[2] = v35;
    if ( (*((_QWORD *)v35 + 1) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_24;
    v36 = *(_QWORD *)(*(_QWORD *)(v2[104] + 136LL) + 8 * v29);
    v12 = __CFADD__(v36, v3);
    v37 = v36 + v3;
    if ( v12 )
      v37 = -1;
    v12 = __CFADD__(v36, v37);
    v3 = v36 + v37;
    if ( v12 )
    {
LABEL_23:
      v3 = -1;
      goto LABEL_24;
    }
  }
  return v3;
}
