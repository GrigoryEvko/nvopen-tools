// Function: sub_23CF510
// Address: 0x23cf510
//
bool __fastcall sub_23CF510(_BYTE *a1, __int64 a2, __int64 a3)
{
  char v4; // r13
  char v5; // al
  unsigned int v6; // r13d
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // rcx
  _BYTE *v11; // rdi
  _BYTE *v12; // r15
  _BYTE *v13; // r14
  __int64 v14; // rdi
  __int64 v15; // r12
  unsigned int v16; // r15d
  unsigned __int64 *v17; // rcx
  __int64 v18; // rcx
  unsigned int v19; // r13d
  bool v20; // al
  __int64 v21; // rdx
  _BYTE *v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // r14
  _BYTE *v25; // rax
  unsigned __int8 *v26; // rcx
  unsigned int v27; // r13d
  int v28; // eax
  unsigned int v29; // r14d
  __int64 v30; // rax
  int v31; // eax
  char v32; // [rsp+0h] [rbp-40h]
  int v33; // [rsp+0h] [rbp-40h]
  int v34; // [rsp+4h] [rbp-3Ch]
  unsigned __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]
  unsigned __int8 *v37; // [rsp+8h] [rbp-38h]

  while ( 1 )
  {
    v4 = *(_BYTE *)(a2 + 24);
    v5 = *a1;
    if ( !v4 )
    {
      if ( v5 != 58 )
        break;
      goto LABEL_11;
    }
    if ( v5 != 57 )
      break;
    v12 = (_BYTE *)*((_QWORD *)a1 - 8);
    v11 = v12;
    if ( !v12 )
      goto LABEL_4;
    v18 = *((_QWORD *)a1 - 4);
    if ( *(_BYTE *)v18 == 17 )
    {
      v19 = *(_DWORD *)(v18 + 32);
      if ( v19 <= 0x40 )
        v20 = *(_QWORD *)(v18 + 24) == 1;
      else
        v20 = v19 - 1 == (unsigned int)sub_C444A0(v18 + 24);
LABEL_30:
      if ( v20 )
        goto LABEL_31;
      goto LABEL_39;
    }
    v24 = *(_QWORD *)(v18 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v24 + 8) - 17 > 1 || *(_BYTE *)v18 > 0x15u )
      goto LABEL_12;
    v36 = *((_QWORD *)a1 - 4);
    v25 = sub_AD7630(v36, 0, a3);
    v26 = (unsigned __int8 *)v36;
    if ( !v25 || *v25 != 17 )
    {
      if ( *(_BYTE *)(v24 + 8) == 17 )
      {
        v28 = *(_DWORD *)(v24 + 32);
        v32 = 0;
        v29 = 0;
        v34 = v28;
        if ( v28 )
        {
          while ( 1 )
          {
            v37 = v26;
            v30 = sub_AD69F0(v26, v29);
            v26 = v37;
            if ( !v30 )
              break;
            if ( *(_BYTE *)v30 != 13 )
            {
              if ( *(_BYTE *)v30 != 17 )
                goto LABEL_39;
              a3 = *(unsigned int *)(v30 + 32);
              if ( (unsigned int)a3 <= 0x40 )
              {
                if ( *(_QWORD *)(v30 + 24) != 1 )
                  goto LABEL_39;
                v32 = v4;
              }
              else
              {
                v33 = *(_DWORD *)(v30 + 32);
                v31 = sub_C444A0(v30 + 24);
                a3 = (unsigned int)(v33 - 1);
                if ( v31 != (_DWORD)a3 )
                  goto LABEL_39;
                v32 = v4;
                v26 = v37;
              }
            }
            if ( v34 == ++v29 )
            {
              if ( v32 )
                goto LABEL_31;
              goto LABEL_39;
            }
          }
        }
      }
      goto LABEL_39;
    }
    v27 = *((_DWORD *)v25 + 8);
    if ( v27 > 0x40 )
    {
      v20 = v27 - 1 == (unsigned int)sub_C444A0((__int64)(v25 + 24));
      goto LABEL_30;
    }
    if ( *((_QWORD *)v25 + 3) == 1 )
    {
LABEL_31:
      *(_BYTE *)(a2 + 25) = 1;
      goto LABEL_32;
    }
LABEL_39:
    v5 = *a1;
    if ( *a1 != 57 )
      break;
LABEL_11:
    v11 = (_BYTE *)*((_QWORD *)a1 - 8);
LABEL_12:
    if ( !v11 )
      goto LABEL_4;
    v12 = (_BYTE *)*((_QWORD *)a1 - 4);
    if ( !v12 )
      goto LABEL_4;
    if ( !(unsigned __int8)sub_23CF510(v11, a2) )
      return 0;
LABEL_32:
    a1 = v12;
  }
  if ( v5 != 55 )
    goto LABEL_4;
  v13 = (_BYTE *)*((_QWORD *)a1 - 8);
  if ( !v13 )
    goto LABEL_4;
  v14 = *((_QWORD *)a1 - 4);
  if ( *(_BYTE *)v14 == 17 )
  {
    v15 = v14 + 24;
    goto LABEL_20;
  }
  v21 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v14 + 8) + 8LL) - 17;
  if ( (unsigned int)v21 > 1 || *(_BYTE *)v14 > 0x15u || (v22 = sub_AD7630(v14, 0, v21)) == 0 || *v22 != 17 )
  {
LABEL_4:
    if ( !*(_QWORD *)a2 )
      *(_QWORD *)a2 = a1;
    v6 = *(_DWORD *)(a2 + 16);
    v7 = 1;
    v8 = 0;
    goto LABEL_7;
  }
  v15 = (__int64)(v22 + 24);
LABEL_20:
  if ( !*(_QWORD *)a2 )
    *(_QWORD *)a2 = v13;
  v16 = *(_DWORD *)(v15 + 8);
  v6 = *(_DWORD *)(a2 + 16);
  if ( v16 > 0x40 )
  {
    v35 = *(unsigned int *)(a2 + 16);
    if ( v16 - (unsigned int)sub_C444A0(v15) <= 0x40 )
    {
      v23 = **(_QWORD **)v15;
      if ( v35 > v23 )
      {
        v8 = **(_QWORD **)v15;
        a1 = v13;
        v7 = 1LL << v23;
        goto LABEL_7;
      }
    }
    return 0;
  }
  v17 = *(unsigned __int64 **)v15;
  if ( (unsigned __int64)*(unsigned int *)(a2 + 16) <= *(_QWORD *)v15 )
    return 0;
  v8 = *(_QWORD *)v15;
  a1 = v13;
  v7 = 1LL << (char)v17;
LABEL_7:
  v9 = *(_QWORD *)(a2 + 8);
  if ( v6 <= 0x40 )
    *(_QWORD *)(a2 + 8) = v9 | v7;
  else
    *(_QWORD *)(v9 + 8LL * (v8 >> 6)) |= v7;
  return *(_QWORD *)a2 == (_QWORD)a1;
}
