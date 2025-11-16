// Function: sub_16B1040
// Address: 0x16b1040
//
void __fastcall sub_16B1040(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v5; // r15
  char v6; // r11
  __int64 v7; // rdx
  unsigned __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r10
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // r13
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rax
  char v17; // bl
  unsigned __int64 v18; // r9
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rax
  char v23; // r12
  char v26; // [rsp+Fh] [rbp-E1h]
  unsigned __int64 v27; // [rsp+10h] [rbp-E0h]
  char v28; // [rsp+10h] [rbp-E0h]
  char v29; // [rsp+10h] [rbp-E0h]
  char v30; // [rsp+10h] [rbp-E0h]
  __int64 v31; // [rsp+18h] [rbp-D8h]
  __int64 v32; // [rsp+18h] [rbp-D8h]
  __int64 v33; // [rsp+18h] [rbp-D8h]
  char v34; // [rsp+18h] [rbp-D8h]
  __int64 v35; // [rsp+18h] [rbp-D8h]
  _BYTE *v36; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+38h] [rbp-B8h]
  _BYTE v38[176]; // [rsp+40h] [rbp-B0h] BYREF

  v5 = a4;
  v36 = v38;
  v37 = 0x8000000000LL;
  if ( !a2 )
    goto LABEL_44;
  v6 = a5;
  v7 = 0;
  v8 = 0;
  v9 = 0x100002600LL;
  v10 = a4;
  while ( 1 )
  {
    if ( !(_DWORD)v7 )
    {
      v12 = v8;
      goto LABEL_13;
    }
    v11 = *(unsigned __int8 *)(a1 + v8);
LABEL_5:
    v12 = v8 + 1;
    if ( (_BYTE)v11 == 92 && v12 < a2 )
    {
      v23 = *(_BYTE *)(a1 + v12);
      if ( HIDWORD(v37) <= (unsigned int)v7 )
      {
        v30 = v6;
        v35 = v10;
        sub_16CD150(&v36, v38, 0, 1);
        v7 = (unsigned int)v37;
        v6 = v30;
        v9 = 0x100002600LL;
        v10 = v35;
      }
      v8 += 2LL;
      v36[v7] = v23;
      v7 = (unsigned int)(v37 + 1);
      LODWORD(v37) = v37 + 1;
      goto LABEL_23;
    }
    if ( (_BYTE)v11 == 34 || (_BYTE)v11 == 39 )
    {
      if ( v12 == a2 )
      {
LABEL_24:
        v5 = v10;
        if ( !(_DWORD)v7 )
          goto LABEL_44;
        goto LABEL_25;
      }
      v26 = v6;
      while ( 2 )
      {
        v17 = *(_BYTE *)(a1 + v12);
        v18 = v12 + 1;
        if ( v17 == (_BYTE)v11 )
        {
          v6 = v26;
          v8 = v12 + 1;
          goto LABEL_23;
        }
        if ( v17 == 92 )
        {
          if ( a2 == v18 )
          {
            if ( HIDWORD(v37) > (unsigned int)v7 )
            {
              v5 = v10;
              v36[v7] = 92;
              v7 = (unsigned int)(v37 + 1);
              LODWORD(v37) = v37 + 1;
              goto LABEL_43;
            }
LABEL_40:
            v27 = v18;
            v31 = v10;
            sub_16CD150(&v36, v38, 0, 1);
            v7 = (unsigned int)v37;
            v9 = 0x100002600LL;
            v10 = v31;
            v12 = v27;
          }
          else
          {
            v17 = *(_BYTE *)(a1 + v18);
            v12 += 2LL;
            if ( HIDWORD(v37) <= (unsigned int)v7 )
            {
LABEL_39:
              v18 = v12;
              goto LABEL_40;
            }
          }
        }
        else
        {
          ++v12;
          if ( HIDWORD(v37) <= (unsigned int)v7 )
            goto LABEL_39;
        }
        v36[v7] = v17;
        v7 = (unsigned int)(v37 + 1);
        LODWORD(v37) = v37 + 1;
        if ( a2 == v12 )
          goto LABEL_24;
        continue;
      }
    }
    if ( (unsigned __int8)v11 <= 0x20u && _bittest64(&v9, v11) )
      break;
    if ( HIDWORD(v37) <= (unsigned int)v7 )
    {
      v29 = v6;
      v33 = v10;
      sub_16CD150(&v36, v38, 0, 1);
      v7 = (unsigned int)v37;
      v6 = v29;
      v9 = 0x100002600LL;
      v10 = v33;
    }
    ++v8;
    v36[v7] = v11;
    v7 = (unsigned int)(v37 + 1);
    LODWORD(v37) = v37 + 1;
LABEL_23:
    if ( a2 == v8 )
      goto LABEL_24;
  }
  if ( (_DWORD)v7 )
  {
    v28 = v6;
    v32 = v10;
    v20 = sub_16D3940(a3, v36, v7);
    v10 = v32;
    v6 = v28;
    v9 = 0x100002600LL;
    v21 = v20;
    v22 = *(unsigned int *)(v32 + 8);
    if ( (unsigned int)v22 >= *(_DWORD *)(v32 + 12) )
    {
      sub_16CD150(v32, v32 + 16, 0, 8);
      v10 = v32;
      v6 = v28;
      v9 = 0x100002600LL;
      v22 = *(unsigned int *)(v32 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v10 + 8 * v22) = v21;
    ++*(_DWORD *)(v10 + 8);
  }
  LODWORD(v37) = 0;
  if ( v12 == a2 )
  {
    v5 = v10;
    goto LABEL_44;
  }
LABEL_13:
  v8 = v12;
  v13 = v10;
  do
  {
    while ( 1 )
    {
      v11 = *(unsigned __int8 *)(a1 + v8);
      if ( (unsigned __int8)v11 > 0x20u || !_bittest64(&v9, v11) )
      {
        v7 = (unsigned int)v37;
        v10 = v13;
        goto LABEL_5;
      }
      if ( (_BYTE)v11 == 10 && v6 )
        break;
      if ( ++v8 == a2 )
        goto LABEL_51;
    }
    v19 = *(unsigned int *)(v13 + 8);
    if ( (unsigned int)v19 >= *(_DWORD *)(v13 + 12) )
    {
      v34 = v6;
      sub_16CD150(v13, v13 + 16, 0, 8);
      v19 = *(unsigned int *)(v13 + 8);
      v6 = v34;
      v9 = 0x100002600LL;
    }
    ++v8;
    *(_QWORD *)(*(_QWORD *)v13 + 8 * v19) = 0;
    ++*(_DWORD *)(v13 + 8);
  }
  while ( v8 != a2 );
LABEL_51:
  v7 = (unsigned int)v37;
  v5 = v13;
LABEL_43:
  if ( !(_DWORD)v7 )
  {
LABEL_44:
    if ( !a5 )
      goto LABEL_45;
    goto LABEL_28;
  }
LABEL_25:
  v14 = sub_16D3940(a3, v36, v7);
  v15 = *(unsigned int *)(v5 + 8);
  if ( (unsigned int)v15 >= *(_DWORD *)(v5 + 12) )
  {
    sub_16CD150(v5, v5 + 16, 0, 8);
    v15 = *(unsigned int *)(v5 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v5 + 8 * v15) = v14;
  ++*(_DWORD *)(v5 + 8);
  if ( a5 )
  {
LABEL_28:
    v16 = *(unsigned int *)(v5 + 8);
    if ( (unsigned int)v16 >= *(_DWORD *)(v5 + 12) )
    {
      sub_16CD150(v5, v5 + 16, 0, 8);
      v16 = *(unsigned int *)(v5 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v5 + 8 * v16) = 0;
    ++*(_DWORD *)(v5 + 8);
  }
LABEL_45:
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
}
