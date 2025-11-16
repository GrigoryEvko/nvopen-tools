// Function: sub_291A4D0
// Address: 0x291a4d0
//
char __fastcall sub_291A4D0(unsigned __int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, _BYTE *a5)
{
  __int64 v8; // rax
  unsigned __int64 v9; // r10
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 v12; // r14
  char v13; // al
  __int64 v14; // rax
  unsigned __int64 v15; // r8
  __int64 v16; // rax
  unsigned int v17; // ecx
  __int64 v18; // rbx
  __int64 v19; // rdi
  unsigned int v20; // r12d
  __int64 v21; // rsi
  int v22; // edx
  unsigned __int64 v23; // rbx
  __int64 v24; // r14
  int v25; // edx
  bool v26; // al
  unsigned __int64 v27; // rdx
  bool v28; // al
  __int64 v29; // rdx
  __int64 v30; // rsi
  unsigned __int64 v32; // [rsp+8h] [rbp-68h]
  unsigned __int64 v33; // [rsp+8h] [rbp-68h]
  unsigned __int64 v34; // [rsp+10h] [rbp-60h]
  unsigned __int64 v35; // [rsp+10h] [rbp-60h]
  unsigned __int64 v36; // [rsp+10h] [rbp-60h]
  unsigned __int64 v37; // [rsp+18h] [rbp-58h]
  unsigned __int64 v38; // [rsp+18h] [rbp-58h]
  unsigned __int64 v40; // [rsp+28h] [rbp-48h]
  unsigned __int64 v41; // [rsp+28h] [rbp-48h]

  v8 = sub_9208B0(a4, a3);
  v9 = a1[1] - a2;
  v10 = a1[2] & 0xFFFFFFFFFFFFFFF8LL;
  v11 = (unsigned __int64)(v8 + 7) >> 3;
  v12 = *(_QWORD *)(v10 + 24);
  v40 = *a1;
  if ( *(_BYTE *)v12 != 85 )
    goto LABEL_2;
  v16 = *(_QWORD *)(v12 - 32);
  if ( !v16 )
    goto LABEL_4;
  if ( !*(_BYTE *)v16 && *(_QWORD *)(v16 + 24) == *(_QWORD *)(v12 + 80) )
  {
    v32 = a1[2] & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_BYTE *)(v16 + 33) & 0x20) != 0 )
    {
      v35 = v11;
      v38 = a1[1] - a2;
      v26 = sub_B46A10(v12);
      v27 = v32;
      if ( v26 || (v33 = v35, v36 = v27, v28 = sub_BD2BE0(v12), v9 = v38, v10 = v36, v11 = v33, v28) )
      {
        LOBYTE(v14) = 1;
        return v14;
      }
LABEL_2:
      if ( v9 > v11 )
        goto LABEL_4;
      v12 = *(_QWORD *)(v10 + 24);
      v13 = *(_BYTE *)v12;
      if ( *(_BYTE *)v12 <= 0x1Cu )
        goto LABEL_4;
      v15 = v40;
      v34 = v9;
      v41 = v11;
      v37 = v15 - a2;
      if ( v13 != 61 )
      {
        if ( v13 != 62 )
        {
          if ( v13 != 85 )
            goto LABEL_4;
          v16 = *(_QWORD *)(v12 - 32);
          if ( !v16 )
            goto LABEL_4;
          goto LABEL_10;
        }
        if ( (*(_BYTE *)(v12 + 2) & 1) != 0 )
          goto LABEL_4;
        v24 = *(_QWORD *)(*(_QWORD *)(v12 - 64) + 8LL);
        if ( sub_9C6480(a4, v24) > v11 || a2 > *a1 )
          goto LABEL_4;
        v25 = *(unsigned __int8 *)(v24 + 8);
        if ( (unsigned int)(v25 - 17) <= 1 || v37 || v34 != v41 )
        {
          if ( (_BYTE)v25 != 12 )
          {
            if ( v37 || v34 != v41 )
              goto LABEL_4;
            goto LABEL_51;
          }
        }
        else
        {
          *a5 = 1;
          if ( *(_BYTE *)(v24 + 8) != 12 )
          {
LABEL_51:
            v29 = a3;
            v30 = v24;
            goto LABEL_47;
          }
        }
        v21 = v24;
        v23 = *(_DWORD *)(v24 + 8) >> 8;
LABEL_30:
        LOBYTE(v14) = v23 >= ((sub_9208B0(a4, v21) + 7) & 0xFFFFFFFFFFFFFFF8LL);
        return v14;
      }
      if ( (*(_BYTE *)(v12 + 2) & 1) != 0 || sub_9C6480(a4, *(_QWORD *)(v12 + 8)) > v11 || a2 > *a1 )
        goto LABEL_4;
      v21 = *(_QWORD *)(v12 + 8);
      v22 = *(unsigned __int8 *)(v21 + 8);
      if ( (unsigned int)(v22 - 17) <= 1 || v37 || v34 != v41 )
      {
        if ( (_BYTE)v22 != 12 )
        {
          if ( v37 || v34 != v41 )
            goto LABEL_4;
          goto LABEL_46;
        }
      }
      else
      {
        *a5 = 1;
        v21 = *(_QWORD *)(v12 + 8);
        if ( *(_BYTE *)(v21 + 8) != 12 )
        {
LABEL_46:
          v29 = v21;
          v30 = a3;
LABEL_47:
          LOBYTE(v14) = sub_29191E0(a4, v30, v29);
          return v14;
        }
      }
      v23 = *(_DWORD *)(v21 + 8) >> 8;
      goto LABEL_30;
    }
  }
  if ( v9 > v11 )
    goto LABEL_4;
LABEL_10:
  if ( *(_BYTE *)v16 )
    goto LABEL_4;
  if ( *(_QWORD *)(v16 + 24) != *(_QWORD *)(v12 + 80) )
    goto LABEL_4;
  if ( (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
    goto LABEL_4;
  v17 = *(_DWORD *)(v16 + 36) - 238;
  if ( v17 > 7 )
    goto LABEL_4;
  LOBYTE(v14) = 0;
  if ( ((1LL << v17) & 0xAD) == 0 )
    return v14;
  v18 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
  v19 = *(_QWORD *)(v12 + 32 * (3 - v18));
  v20 = *(_DWORD *)(v19 + 32);
  if ( v20 <= 0x40 )
  {
    if ( *(_QWORD *)(v19 + 24) )
      goto LABEL_4;
  }
  else if ( v20 != (unsigned int)sub_C444A0(v19 + 24) )
  {
LABEL_4:
    LOBYTE(v14) = 0;
    return v14;
  }
  if ( **(_BYTE **)(v12 + 32 * (2 - v18)) > 0x15u )
    goto LABEL_4;
  return ((__int64)a1[2] >> 2) & 1;
}
