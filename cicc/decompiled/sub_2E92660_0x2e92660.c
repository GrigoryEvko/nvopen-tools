// Function: sub_2E92660
// Address: 0x2e92660
//
unsigned __int64 __fastcall sub_2E92660(unsigned __int64 a1, unsigned int a2, __int64 a3)
{
  unsigned int v4; // ecx
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // r12
  char v10; // r8
  unsigned int v11; // r15d
  char v12; // r11
  __int64 v13; // rax
  __int64 v14; // rdx
  bool v15; // cl
  char v16; // al
  __int16 *v17; // rax
  __int16 *v18; // rdx
  __int16 v19; // si
  unsigned __int16 v20; // ax
  __int16 v21; // di
  unsigned __int8 v22; // al
  char v23; // si
  bool v24; // zf
  char v25; // al
  unsigned __int16 v26; // ax
  char v28; // di
  char v29; // di
  __int64 v30; // [rsp+0h] [rbp-70h]
  char v31; // [rsp+Fh] [rbp-61h]
  __int64 v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+18h] [rbp-58h]
  unsigned __int8 v34; // [rsp+20h] [rbp-50h]
  unsigned int v35; // [rsp+28h] [rbp-48h]
  unsigned __int8 v36; // [rsp+2Ch] [rbp-44h]
  char v37; // [rsp+2Dh] [rbp-43h]
  unsigned __int8 v38; // [rsp+2Eh] [rbp-42h]
  unsigned __int8 v39; // [rsp+2Fh] [rbp-41h]
  __int64 i; // [rsp+30h] [rbp-40h]
  __int64 v41; // [rsp+38h] [rbp-38h]

  v4 = a2;
  v5 = a1;
  v6 = *(_QWORD *)(a1 + 24);
  v7 = v6 + 48;
  for ( i = *(_QWORD *)(*(_QWORD *)(v6 + 56) + 32LL) + 40LL * (*(_DWORD *)(*(_QWORD *)(v6 + 56) + 40LL) & 0xFFFFFF);
        (*(_BYTE *)(v5 + 44) & 4) != 0;
        v5 = *(_QWORD *)v5 & 0xFFFFFFFFFFFFFFF8LL )
  {
    ;
  }
  while ( 1 )
  {
    v8 = *(_QWORD *)(v5 + 32);
    v9 = v8 + 40LL * (*(_DWORD *)(v5 + 40) & 0xFFFFFF);
    if ( v8 != v9 )
      break;
    v5 = *(_QWORD *)(v5 + 8);
    if ( v7 == v5 )
      break;
    if ( (*(_BYTE *)(v5 + 44) & 4) == 0 )
    {
      v5 = v6 + 48;
      break;
    }
  }
  LOBYTE(v33) = 0;
  v10 = 0;
  v36 = 0;
  v38 = 0;
  v30 = 24LL * a2;
  v39 = 0;
  v34 = 0;
  v41 = 4LL * (a2 >> 5);
  v37 = 1;
  if ( v5 == v7 )
    goto LABEL_20;
  do
  {
    do
    {
      if ( *(_BYTE *)v8 == 12 )
      {
        if ( ((*(_DWORD *)(*(_QWORD *)(v8 + 24) + v41) >> v4) & 1) == 0 )
          v10 = 1;
LABEL_13:
        v8 += 40;
        v13 = v9;
        if ( v8 == v9 )
          goto LABEL_17;
LABEL_37:
        v9 = v13;
        continue;
      }
      if ( *(_BYTE *)v8 )
        goto LABEL_13;
      v11 = *(_DWORD *)(v8 + 8);
      if ( v11 - 1 > 0x3FFFFFFE )
        goto LABEL_13;
      v12 = 1;
      if ( v11 != v4 )
      {
        if ( v4 - 1 > 0x3FFFFFFE )
          goto LABEL_13;
        v31 = v10;
        v35 = v4;
        v32 = a3;
        v16 = sub_E92070(a3, v11, v4);
        a3 = v32;
        v4 = v35;
        v10 = v31;
        v12 = v16;
        if ( !v16 )
          goto LABEL_13;
        v17 = (__int16 *)(*(_QWORD *)(v32 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v32 + 8) + v30 + 8));
        v18 = v17 + 1;
        v19 = *v17;
        v20 = v35 + *v17;
        if ( v19 )
        {
          while ( v11 != v20 )
          {
            v21 = *v18++;
            v20 += v21;
            if ( !v21 )
              goto LABEL_29;
          }
        }
        else
        {
LABEL_29:
          v12 = 0;
        }
      }
      v22 = *(_BYTE *)(v8 + 3);
      v23 = v22 & 0x10;
      if ( (*(_BYTE *)(v8 + 4) & 1) != 0 || (*(_BYTE *)(v8 + 4) & 2) != 0 )
      {
        if ( v23 )
        {
LABEL_45:
          v28 = v39;
          v34 = 1;
          if ( v12 )
            v28 = 1;
          v39 = v28;
          v29 = v37;
          if ( (((v22 & 0x10) != 0) & (v22 >> 6)) == 0 )
            v29 = 0;
          v37 = v29;
        }
        goto LABEL_13;
      }
      if ( v23 && (*(_DWORD *)v8 & 0xFFF00) == 0 )
        goto LABEL_45;
      v38 = 1;
      if ( !v12 )
        goto LABEL_13;
      v36 = 1;
      v24 = ((v23 == 0) & (v22 >> 6)) == 0;
      v25 = v33;
      if ( !v24 )
        v25 = 1;
      v8 += 40;
      LOBYTE(v33) = v25;
      v13 = v9;
      if ( v8 != v9 )
        goto LABEL_37;
      while ( 1 )
      {
LABEL_17:
        v5 = *(_QWORD *)(v5 + 8);
        if ( v7 == v5 )
        {
LABEL_18:
          v8 = v9;
          v9 = v13;
          goto LABEL_19;
        }
        if ( (*(_BYTE *)(v5 + 44) & 4) == 0 )
          break;
        v9 = *(_QWORD *)(v5 + 32);
        v13 = v9 + 40LL * (*(_DWORD *)(v5 + 40) & 0xFFFFFF);
        if ( v9 != v13 )
          goto LABEL_18;
      }
      v8 = v9;
      v5 = v7;
      v9 = v13;
LABEL_19:
      ;
    }
    while ( v5 != v7 );
LABEL_20:
    ;
  }
  while ( i != v8 && v9 != v8 );
  if ( v37 )
  {
    if ( v39 )
    {
      v14 = 0;
      v15 = 1;
    }
    else
    {
      v14 = 0;
      if ( !v10 )
        v14 = v34;
      v15 = v10 != 0;
    }
  }
  else
  {
    v14 = 0;
    v15 = 0;
  }
  LOBYTE(v26) = v10;
  HIBYTE(v26) = v34;
  return (v33 << 56)
       | (v14 << 48) & 0xFFFFFFFFFFFFFFLL
       | ((unsigned __int64)v15 << 40) & 0xFFFFFFFFFFFFLL
       | ((unsigned __int64)v36 << 32) & 0xFFFFFFFFFFLL
       | (v38 << 24)
       | ((unsigned __int64)v39 << 16) & 0xFFFFFF
       | v26;
}
