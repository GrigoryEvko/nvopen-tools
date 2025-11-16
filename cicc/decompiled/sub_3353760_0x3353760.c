// Function: sub_3353760
// Address: 0x3353760
//
bool __fastcall sub_3353760(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  unsigned int v6; // r14d
  __int64 v7; // rcx
  __int64 v8; // r8
  char v9; // r9
  unsigned int v10; // r15d
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // r9
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned int v20; // eax
  __int64 v21; // rcx
  __int64 v22; // r8
  bool v23; // cf
  __int64 *v24; // rdx
  __int64 *v25; // rsi
  __int64 v26; // r9
  __int64 *v27; // rdi
  __int64 *v28; // rdx
  bool result; // al
  unsigned __int8 v30; // dl
  unsigned __int8 v31; // al
  unsigned int v32; // edx
  unsigned int v33; // esi
  bool v34; // cc
  __int64 v35; // rdx
  int v36; // r12d
  unsigned int v37; // r12d
  int v38; // r12d
  unsigned int v39; // r12d
  int v40; // eax
  unsigned int v41; // [rsp+Ch] [rbp-34h]

  v5 = a2;
  if ( !byte_5038B88 )
  {
    v30 = (*(_BYTE *)(a1 + 248) & 0x40) != 0;
    v31 = (*(_BYTE *)(a2 + 248) & 0x40) != 0;
    v23 = v30 < v31;
    if ( v30 != v31 )
      return v23;
  }
  v6 = sub_3351730(a3, (unsigned int *)a1);
  v10 = sub_3351730(a3, (unsigned int *)a2);
  v11 = (v8 & 2) != 0;
  if ( (v8 & 2) != 0 && (v9 & 4) != 0 )
  {
    v7 = 0;
    v12 = *(_DWORD *)(*(_QWORD *)a2 + 68LL);
    a2 = v10 - v12;
    if ( v10 > v12 )
      v7 = (unsigned int)a2;
    v10 = v7;
  }
  v13 = v9 & 2;
  if ( (_DWORD)v13 )
  {
    v8 &= 4u;
    if ( (_DWORD)v8 )
    {
      v14 = *(_QWORD *)a1;
      v32 = *(_DWORD *)(*(_QWORD *)a1 + 68LL);
      v33 = v6 - v32;
      v34 = v6 <= v32;
      v6 = 0;
      if ( !v34 )
        v6 = v33;
      if ( v10 == v6 )
      {
        a2 = *(_QWORD *)v5;
        goto LABEL_11;
      }
    }
    else if ( v10 == v6 )
    {
      goto LABEL_10;
    }
    return v10 < v6;
  }
  if ( v10 != v6 )
    return v10 < v6;
  if ( (v8 & 2) == 0 )
    goto LABEL_15;
LABEL_10:
  v14 = *(_QWORD *)a1;
  a2 = *(_QWORD *)v5;
  if ( !*(_QWORD *)a1 )
  {
    if ( !a2 )
      goto LABEL_15;
    v11 = 0;
    goto LABEL_12;
  }
LABEL_11:
  v11 = *(unsigned int *)(v14 + 72);
  if ( !a2 )
  {
    v15 = *(_DWORD *)(v14 + 72);
    a2 = 0;
    goto LABEL_13;
  }
LABEL_12:
  a2 = *(unsigned int *)(a2 + 72);
  v15 = v11 | a2;
LABEL_13:
  if ( !v15 || (_DWORD)a2 == (_DWORD)v11 )
  {
LABEL_15:
    v41 = sub_3351600(a1, (_QWORD *)a2, v11, v7, v8, v13);
    v20 = sub_3351600(v5, (_QWORD *)a2, v16, v17, v18, v19);
    v23 = v41 < v20;
    if ( v41 != v20 )
      return v23;
    v24 = *(__int64 **)(a1 + 40);
    v25 = *(__int64 **)(v5 + 40);
    v26 = (__int64)&v24[2 * *(unsigned int *)(a1 + 48)];
    v27 = &v25[2 * *(unsigned int *)(v5 + 48)];
    if ( v24 == (__int64 *)v26 )
    {
      if ( v25 == v27 )
      {
LABEL_23:
        if ( !v10 || (*(_BYTE *)(a1 + 248) & 2) == 0 )
        {
          v35 = (*(_BYTE *)(v5 + 248) & 2) != 0;
          if ( !v6 || (*(_BYTE *)(v5 + 248) & 2) == 0 )
          {
            LOBYTE(v35) = byte_5038F08 | v35;
            if ( (_BYTE)v35 || (*(_BYTE *)(a1 + 248) & 2) != 0 )
            {
              if ( (*(_BYTE *)(a1 + 254) & 2) == 0 )
                sub_2F8F770(a1, v25, v35, v21, v22, v26);
              v36 = *(_DWORD *)(a1 + 244);
              if ( (*(_BYTE *)(v5 + 254) & 2) == 0 )
                sub_2F8F770(v5, v25, v35, v21, v22, v26);
              if ( *(_DWORD *)(v5 + 244) != v36 )
              {
                if ( (*(_BYTE *)(a1 + 254) & 2) == 0 )
                  sub_2F8F770(a1, v25, v35, v21, v22, v26);
                v37 = *(_DWORD *)(a1 + 244);
                if ( (*(_BYTE *)(v5 + 254) & 2) == 0 )
                  sub_2F8F770(v5, v25, v35, v21, v22, v26);
                return *(_DWORD *)(v5 + 244) < v37;
              }
              if ( (*(_BYTE *)(a1 + 254) & 1) == 0 )
                sub_2F8F5D0(a1, v25, v35, v21, v22, v26);
              v38 = *(_DWORD *)(a1 + 240);
              if ( (*(_BYTE *)(v5 + 254) & 1) == 0 )
                sub_2F8F5D0(v5, v25, v35, v21, v22, v26);
              if ( *(_DWORD *)(v5 + 240) != v38 )
              {
                if ( (*(_BYTE *)(a1 + 254) & 1) == 0 )
                  sub_2F8F5D0(a1, v25, v35, v21, v22, v26);
                v39 = *(_DWORD *)(a1 + 240);
                if ( (*(_BYTE *)(v5 + 254) & 1) == 0 )
                  sub_2F8F5D0(v5, v25, v35, v21, v22, v26);
                return *(_DWORD *)(v5 + 240) > v39;
              }
            }
            else
            {
              v40 = sub_33532E0(a1, v5, 0, a3, v22, v26);
              if ( v40 )
                return v40 > 0;
            }
          }
        }
        return *(_DWORD *)(a1 + 204) > *(_DWORD *)(v5 + 204);
      }
      v22 = 0;
    }
    else
    {
      LODWORD(v22) = 0;
      do
      {
        v22 = (((*v24 >> 1) & 3) == 0) + (unsigned int)v22;
        v24 += 2;
      }
      while ( (__int64 *)v26 != v24 );
      if ( v25 == v27 )
      {
        v25 = 0;
        goto LABEL_22;
      }
    }
    v28 = *(__int64 **)(v5 + 40);
    LODWORD(v25) = 0;
    do
    {
      v25 = (__int64 *)((((*v28 >> 1) & 3) == 0) + (unsigned int)v25);
      v28 += 2;
    }
    while ( v27 != v28 );
LABEL_22:
    v23 = (unsigned int)v25 < (unsigned int)v22;
    if ( (_DWORD)v25 == (_DWORD)v22 )
      goto LABEL_23;
    return v23;
  }
  result = 0;
  if ( (_DWORD)v11 )
    return (_DWORD)a2 == 0 || (unsigned int)a2 > (unsigned int)v11;
  return result;
}
