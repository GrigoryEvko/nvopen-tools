// Function: sub_98FF80
// Address: 0x98ff80
//
__int64 __fastcall sub_98FF80(__int64 a1, unsigned int a2, unsigned __int8 *a3)
{
  unsigned int v5; // ebx
  __int64 v6; // r13
  int v7; // eax
  bool v8; // r12
  unsigned int v9; // r15d
  __int64 *v10; // rdi
  __int64 v11; // rsi
  unsigned int v12; // ecx
  int v13; // ebx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned int v17; // ecx
  unsigned int v18; // ebx
  int v19; // r15d
  unsigned __int8 *v20; // rax
  unsigned __int8 *v21; // rdx
  int v22; // ecx
  unsigned int v23; // r8d
  unsigned __int8 *v24; // rdi
  __int64 v25; // r9
  unsigned int v26; // ecx
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  unsigned int v31; // r8d
  unsigned int v32; // ecx
  char v33; // al
  __int64 v34; // r9
  unsigned int v35; // ecx
  int v36; // eax
  int v37; // eax
  bool v38; // al
  int v39; // eax
  char v40; // al
  unsigned __int8 *v41; // [rsp+0h] [rbp-60h]
  unsigned __int8 *v42; // [rsp+0h] [rbp-60h]
  int v43; // [rsp+0h] [rbp-60h]
  int v44; // [rsp+0h] [rbp-60h]
  unsigned int v45; // [rsp+8h] [rbp-58h]
  unsigned int v46; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v47; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v48; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v49; // [rsp+18h] [rbp-48h]
  __int64 v50; // [rsp+20h] [rbp-40h]
  char v51; // [rsp+28h] [rbp-38h]
  __int64 v52; // [rsp+28h] [rbp-38h]

  if ( (unsigned int)*a3 - 12 <= 1 )
    goto LABEL_2;
  v5 = a2;
  v6 = (__int64)a3;
  v50 = *((_QWORD *)a3 + 1);
  v51 = sub_B532B0(a2);
  v7 = sub_B52EF0(a2);
  v8 = v7 == 34 || v7 == 37;
  if ( *(_BYTE *)v6 == 17 )
  {
    v9 = *(_DWORD *)(v6 + 32);
    v10 = (__int64 *)(v6 + 24);
    if ( v8 )
    {
      if ( v51 )
      {
        v11 = *(_QWORD *)(v6 + 24);
        v12 = v9 - 1;
        if ( v9 > 0x40 )
        {
          if ( (*(_QWORD *)(v11 + 8LL * (v12 >> 6)) & (1LL << v12)) != 0 || (unsigned int)sub_C445E0(v10) != v12 )
            goto LABEL_10;
LABEL_2:
          *(_BYTE *)(a1 + 16) = 0;
          return a1;
        }
        v33 = (1LL << v12) - 1 != v11;
LABEL_47:
        if ( v33 )
          goto LABEL_10;
        goto LABEL_2;
      }
      if ( !v9 )
        goto LABEL_2;
      if ( v9 <= 0x40 )
      {
        v33 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) != *(_QWORD *)(v6 + 24);
        goto LABEL_47;
      }
LABEL_46:
      v33 = v9 != (unsigned int)sub_C445E0(v10);
      goto LABEL_47;
    }
    if ( !v51 )
    {
      if ( v9 <= 0x40 )
        v33 = *(_QWORD *)(v6 + 24) != 0;
      else
        v33 = v9 != (unsigned int)sub_C444A0(v10);
      goto LABEL_47;
    }
    v16 = *(_QWORD *)(v6 + 24);
    v17 = v9 - 1;
    if ( v9 <= 0x40 )
    {
      v33 = 1LL << v17 != v16;
      goto LABEL_47;
    }
    if ( (*(_QWORD *)(v16 + 8LL * (v17 >> 6)) & (1LL << v17)) != 0 && (unsigned int)sub_C44590(v10) == v17 )
      goto LABEL_2;
    goto LABEL_10;
  }
  if ( *(_BYTE *)(v50 + 8) == 17 )
  {
    if ( !*(_DWORD *)(v50 + 32) )
      goto LABEL_10;
    v49 = 0;
    v18 = 0;
    v19 = *(_DWORD *)(v50 + 32);
    while ( 1 )
    {
      v20 = (unsigned __int8 *)sub_AD69F0(v6, v18);
      v21 = v20;
      if ( !v20 )
        goto LABEL_2;
      v22 = *v20;
      if ( (unsigned int)(v22 - 12) > 1 )
        break;
LABEL_24:
      if ( v19 == ++v18 )
      {
        v5 = a2;
        goto LABEL_11;
      }
    }
    if ( (_BYTE)v22 != 17 )
      goto LABEL_2;
    v23 = *((_DWORD *)v20 + 8);
    v24 = v20 + 24;
    if ( v8 )
    {
      if ( v51 )
      {
        v25 = *((_QWORD *)v20 + 3);
        v26 = v23 - 1;
        if ( v23 > 0x40 )
        {
          v45 = v23 - 1;
          if ( (*(_QWORD *)(v25 + 8LL * (v26 >> 6)) & (1LL << v26)) == 0 )
          {
            v41 = v20;
            v27 = sub_C445E0(v24);
            v21 = v41;
            if ( v45 == v27 )
              goto LABEL_2;
          }
          goto LABEL_53;
        }
        v38 = (1LL << v26) - 1 != v25;
      }
      else
      {
        if ( !v23 )
          goto LABEL_2;
        if ( v23 <= 0x40 )
        {
          v38 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v23) != *((_QWORD *)v20 + 3);
        }
        else
        {
          v44 = *((_DWORD *)v20 + 8);
          v48 = v20;
          v39 = sub_C445E0(v24);
          v21 = v48;
          v38 = v44 != v39;
        }
      }
    }
    else if ( v51 )
    {
      v34 = *((_QWORD *)v20 + 3);
      v35 = v23 - 1;
      if ( v23 > 0x40 )
      {
        v46 = v23 - 1;
        if ( (*(_QWORD *)(v34 + 8LL * (v35 >> 6)) & (1LL << v35)) != 0 )
        {
          v42 = v20;
          v36 = sub_C44590(v24);
          v21 = v42;
          if ( v36 == v46 )
            goto LABEL_2;
        }
        goto LABEL_53;
      }
      v38 = 1LL << v35 != v34;
    }
    else if ( v23 <= 0x40 )
    {
      v38 = *((_QWORD *)v20 + 3) != 0;
    }
    else
    {
      v43 = *((_DWORD *)v20 + 8);
      v47 = v20;
      v37 = sub_C444A0(v24);
      v21 = v47;
      v38 = v43 != v37;
    }
    if ( !v38 )
      goto LABEL_2;
LABEL_53:
    if ( v49 )
      v21 = v49;
    v49 = v21;
    goto LABEL_24;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17 > 1 )
    goto LABEL_2;
  v28 = sub_AD7630(v6, 0);
  if ( !v28 || *(_BYTE *)v28 != 17 )
    goto LABEL_2;
  v10 = (__int64 *)(v28 + 24);
  if ( !v8 )
  {
    if ( v51 )
      v40 = sub_986B30(v10, 0, v29, v30, v31);
    else
      v40 = sub_9867B0((__int64)v10);
    v33 = v40 ^ 1;
    goto LABEL_47;
  }
  v9 = *(_DWORD *)(v28 + 32);
  if ( !v51 )
  {
    if ( !v9 )
      goto LABEL_2;
    if ( v9 <= 0x40 )
    {
      v33 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) != *(_QWORD *)(v28 + 24);
      goto LABEL_47;
    }
    goto LABEL_46;
  }
  v32 = v9 - 1;
  if ( v9 <= 0x40 )
  {
    v33 = (1LL << v32) - 1 != *(_QWORD *)(v28 + 24);
    goto LABEL_47;
  }
  v52 = v28 + 24;
  if ( !sub_986C60(v10, v32) && v9 - 1 == (unsigned int)sub_C445E0(v52) )
    goto LABEL_2;
LABEL_10:
  v49 = 0;
LABEL_11:
  if ( (unsigned __int8)sub_AD6C40(v6) )
    v6 = sub_AD6D90(v6, v49);
  v13 = sub_B53250(v5);
  v14 = sub_AD64C0(v50, -(__int64)!v8 | 1, 1);
  v15 = sub_AD57C0(v6, v14, 0, 0);
  *(_DWORD *)a1 = v13;
  *(_BYTE *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 8) = v15;
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
