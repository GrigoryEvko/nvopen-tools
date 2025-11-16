// Function: sub_10D0230
// Address: 0x10d0230
//
__int64 __fastcall sub_10D0230(__int64 a1, int a2, unsigned __int8 *a3)
{
  unsigned int v3; // r13d
  __int64 v5; // rax
  char *v7; // r12
  char v8; // al
  __int64 v9; // rdx
  _BYTE *v10; // rbx
  unsigned __int64 v11; // rax
  char v12; // dl
  __int64 v13; // rax
  __int64 v14; // rcx
  unsigned __int64 v15; // rax
  unsigned int v16; // edx
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned int v19; // r15d
  int v20; // eax
  bool v21; // al
  __int64 *v22; // rax
  __int64 v23; // r15
  __int64 v24; // rdx
  _BYTE *v25; // rax
  unsigned int v26; // r8d
  unsigned int v27; // r15d
  int v28; // eax
  int v29; // eax
  bool v30; // r15
  __int64 v31; // rax
  unsigned int v32; // r8d
  unsigned int v33; // r15d
  int v34; // eax
  unsigned int v35; // [rsp+Ch] [rbp-54h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+18h] [rbp-48h]
  int v39; // [rsp+18h] [rbp-48h]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v5 != 93 )
    goto LABEL_5;
  if ( *(_DWORD *)(v5 + 80) != 1 )
    goto LABEL_5;
  if ( **(_DWORD **)(v5 + 72) != 1 )
    goto LABEL_5;
  v14 = *(_QWORD *)(v5 - 32);
  if ( !v14 )
    goto LABEL_5;
  **(_QWORD **)a1 = v14;
  **(_QWORD **)(a1 + 8) = v5;
  v7 = (char *)*((_QWORD *)a3 - 4);
  v8 = *v7;
  if ( *v7 != 82 )
    goto LABEL_6;
  v15 = sub_B53900(*((_QWORD *)a3 - 4));
  sub_B53630(v15, *(_QWORD *)(a1 + 16));
  v3 = v16;
  if ( !(_BYTE)v16 )
    goto LABEL_5;
  v17 = *((_QWORD *)v7 - 8);
  if ( *(_BYTE *)v17 != 93
    || *(_DWORD *)(v17 + 80) != 1
    || **(_DWORD **)(v17 + 72)
    || *(_QWORD *)(v17 - 32) != **(_QWORD **)(a1 + 24) )
  {
    goto LABEL_5;
  }
  **(_QWORD **)(a1 + 32) = v17;
  v18 = *((_QWORD *)v7 - 4);
  if ( *(_BYTE *)v18 == 17 )
  {
    v19 = *(_DWORD *)(v18 + 32);
    if ( v19 <= 0x40 )
    {
      v21 = *(_QWORD *)(v18 + 24) == 0;
    }
    else
    {
      v38 = *((_QWORD *)v7 - 4);
      v20 = sub_C444A0(v18 + 24);
      v18 = v38;
      v21 = v19 == v20;
    }
  }
  else
  {
    v23 = *(_QWORD *)(v18 + 8);
    v24 = (unsigned int)*(unsigned __int8 *)(v23 + 8) - 17;
    if ( (unsigned int)v24 > 1 || *(_BYTE *)v18 > 0x15u )
      goto LABEL_5;
    v36 = *((_QWORD *)v7 - 4);
    v25 = sub_AD7630(v18, 0, v24);
    v18 = v36;
    v26 = 0;
    if ( !v25 || *v25 != 17 )
    {
      if ( *(_BYTE *)(v23 + 8) == 17 )
      {
        v29 = *(_DWORD *)(v23 + 32);
        v30 = 0;
        v39 = v29;
        while ( v39 != v26 )
        {
          v35 = v26;
          v37 = v18;
          v31 = sub_AD69F0((unsigned __int8 *)v18, v26);
          if ( !v31 )
            goto LABEL_5;
          v18 = v37;
          v32 = v35;
          if ( *(_BYTE *)v31 != 13 )
          {
            if ( *(_BYTE *)v31 != 17 )
              goto LABEL_5;
            v33 = *(_DWORD *)(v31 + 32);
            if ( v33 <= 0x40 )
            {
              v30 = *(_QWORD *)(v31 + 24) == 0;
            }
            else
            {
              v34 = sub_C444A0(v31 + 24);
              v18 = v37;
              v32 = v35;
              v30 = v33 == v34;
            }
            if ( !v30 )
              goto LABEL_5;
          }
          v26 = v32 + 1;
        }
        if ( v30 )
          goto LABEL_31;
      }
      goto LABEL_5;
    }
    v27 = *((_DWORD *)v25 + 8);
    if ( v27 <= 0x40 )
    {
      v21 = *((_QWORD *)v25 + 3) == 0;
    }
    else
    {
      v28 = sub_C444A0((__int64)(v25 + 24));
      v18 = v36;
      v21 = v27 == v28;
    }
  }
  if ( !v21 )
  {
LABEL_5:
    v7 = (char *)*((_QWORD *)a3 - 4);
    v8 = *v7;
LABEL_6:
    if ( v8 == 93 && *((_DWORD *)v7 + 20) == 1 && **((_DWORD **)v7 + 9) == 1 )
    {
      v9 = *((_QWORD *)v7 - 4);
      if ( v9 )
      {
        **(_QWORD **)a1 = v9;
        **(_QWORD **)(a1 + 8) = v7;
        v10 = (_BYTE *)*((_QWORD *)a3 - 8);
        if ( *v10 == 82 )
        {
          v11 = sub_B53900((__int64)v10);
          sub_B53630(v11, *(_QWORD *)(a1 + 16));
          if ( v12 )
          {
            v13 = *((_QWORD *)v10 - 8);
            if ( *(_BYTE *)v13 == 93
              && *(_DWORD *)(v13 + 80) == 1
              && !**(_DWORD **)(v13 + 72)
              && *(_QWORD *)(v13 - 32) == **(_QWORD **)(a1 + 24) )
            {
              **(_QWORD **)(a1 + 32) = v13;
              v3 = sub_10081F0((__int64 **)(a1 + 40), *((_QWORD *)v10 - 4));
              if ( (_BYTE)v3 )
              {
                **(_QWORD **)(a1 + 48) = v10;
                return v3;
              }
            }
          }
        }
      }
    }
    return 0;
  }
LABEL_31:
  v22 = *(__int64 **)(a1 + 40);
  if ( v22 )
    *v22 = v18;
  **(_QWORD **)(a1 + 48) = v7;
  return v3;
}
