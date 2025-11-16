// Function: sub_996650
// Address: 0x996650
//
char __fastcall sub_996650(__int64 *a1, __int64 a2, unsigned int a3, unsigned __int8 a4, __int64 a5, __int64 a6)
{
  unsigned int *v10; // rax
  unsigned int v11; // eax
  unsigned __int8 v12; // al
  __int64 v13; // rdi
  unsigned int *v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rbx
  unsigned int v18; // r12d
  __int64 v19; // rdx
  int v20; // eax
  int v21; // eax
  __int64 *v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // r10
  __int64 v25; // rdi
  _BYTE *v26; // rdi
  __int64 v27; // r15
  __int64 v28; // rdi
  __int64 v29; // rbx
  __int64 *v30; // rcx
  __int64 v31; // r9
  __int64 v32; // r8
  int v33; // edi
  int v34; // esi
  __int64 v35; // rax
  int *v36; // rdx
  _BYTE *v37; // rdi
  _BYTE *v38; // rax
  __int64 v40; // [rsp+0h] [rbp-A0h]
  __int64 v41; // [rsp+8h] [rbp-98h]
  __int64 v42; // [rsp+8h] [rbp-98h]
  __int64 v45; // [rsp+28h] [rbp-78h]
  __int64 v46; // [rsp+38h] [rbp-68h] BYREF
  __int64 v47; // [rsp+40h] [rbp-60h] BYREF
  int v48; // [rsp+48h] [rbp-58h] BYREF
  char v49; // [rsp+4Ch] [rbp-54h]
  int *v50; // [rsp+50h] [rbp-50h] BYREF
  __int64 *v51; // [rsp+58h] [rbp-48h]
  __int64 *v52; // [rsp+60h] [rbp-40h] BYREF
  char v53; // [rsp+68h] [rbp-38h]

  v10 = (unsigned int *)sub_C94E20(qword_4F862D0);
  if ( v10 )
    v11 = *v10;
  else
    v11 = qword_4F862D0[2];
  if ( a3 >= v11 )
    goto LABEL_12;
  v12 = *(_BYTE *)a2;
  if ( a4 )
  {
    if ( v12 <= 0x1Cu )
      goto LABEL_12;
    v13 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v13 + 8) - 17 <= 1 )
      v13 = **(_QWORD **)(v13 + 16);
    if ( !(unsigned __int8)sub_BCAC40(v13, 1) )
      goto LABEL_12;
    if ( *(_BYTE *)a2 != 57 )
    {
      if ( *(_BYTE *)a2 != 86 )
        goto LABEL_12;
      v41 = *(_QWORD *)(a2 - 96);
      if ( *(_QWORD *)(v41 + 8) != *(_QWORD *)(a2 + 8) )
        goto LABEL_12;
      v37 = *(_BYTE **)(a2 - 32);
      if ( *v37 > 0x15u )
        goto LABEL_12;
      v40 = *(_QWORD *)(a2 - 64);
      if ( !(unsigned __int8)sub_AC30F0(v37) )
        goto LABEL_12;
LABEL_78:
      v23 = v41;
      v24 = v40;
      v46 = v41;
      if ( !v40 )
        goto LABEL_12;
LABEL_34:
      v42 = v24;
      sub_996650(a1, v23, a3 + 1, a4, a5, a6);
      LOBYTE(v16) = sub_996650(a1, v42, a3 + 1, a4, a5, a6);
      return v16;
    }
  }
  else
  {
    if ( v12 <= 0x1Cu )
      goto LABEL_12;
    v25 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v25 + 8) - 17 <= 1 )
      v25 = **(_QWORD **)(v25 + 16);
    if ( !(unsigned __int8)sub_BCAC40(v25, 1) )
      goto LABEL_12;
    if ( *(_BYTE *)a2 != 58 )
    {
      if ( *(_BYTE *)a2 != 86 )
        goto LABEL_12;
      v41 = *(_QWORD *)(a2 - 96);
      if ( *(_QWORD *)(v41 + 8) != *(_QWORD *)(a2 + 8) )
        goto LABEL_12;
      v26 = *(_BYTE **)(a2 - 64);
      if ( *v26 > 0x15u )
        goto LABEL_12;
      v40 = *(_QWORD *)(a2 - 32);
      if ( !(unsigned __int8)sub_AD7A80(v26) )
        goto LABEL_12;
      goto LABEL_78;
    }
  }
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v22 = *(__int64 **)(a2 - 8);
  else
    v22 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v23 = *v22;
  if ( *v22 )
  {
    v24 = v22[4];
    v46 = *v22;
    if ( v24 )
      goto LABEL_34;
  }
LABEL_12:
  v14 = (unsigned int *)sub_C94E20(qword_4F862D0);
  if ( v14 )
    v15 = *v14;
  else
    v15 = qword_4F862D0[2];
  if ( a3 < v15 )
  {
    v50 = 0;
    v51 = &v46;
    if ( (unsigned __int8)sub_996420((_QWORD **)&v50, 30, (unsigned __int8 *)a2) )
    {
      LOBYTE(v16) = sub_996650(a1, v46, a3 + 1, a4 ^ 1u, a5, a6);
      return v16;
    }
  }
  LOBYTE(v16) = *(_BYTE *)a2;
  v48 = 42;
  v49 = 0;
  if ( (unsigned __int8)v16 <= 0x1Cu )
    return v16;
  if ( (_BYTE)v16 != 83 )
  {
    if ( (_BYTE)v16 == 85 )
      goto LABEL_19;
    v51 = a1;
    v50 = &v48;
    v52 = &v47;
    v53 = 0;
    goto LABEL_60;
  }
  v27 = *(_QWORD *)(a2 - 64);
  if ( !v27 )
    return v16;
  v28 = *(_QWORD *)(a2 - 32);
  LOBYTE(v16) = *(_BYTE *)v28;
  v29 = v28 + 24;
  if ( *(_BYTE *)v28 == 18 )
  {
LABEL_51:
    v45 = sub_B53900(a2);
    v48 = v45;
    v49 = BYTE4(v45);
    LOBYTE(v16) = sub_989E10(
                    (__int64)&v50,
                    (unsigned int)v45,
                    *(_QWORD *)(*(_QWORD *)(a5 + 40) + 72LL),
                    v27,
                    v29,
                    a1 != (__int64 *)v27);
    if ( v51 != a1 )
      return v16;
    if ( a4 )
      v21 = HIDWORD(v50);
    else
      v21 = (int)v50;
LABEL_54:
    LODWORD(v16) = *(_DWORD *)a6 & v21;
    *(_DWORD *)a6 = v16 & 0x3FF;
    if ( (v16 & 3) == 0 && !*(_BYTE *)(a6 + 5) )
    {
      if ( (v16 & 0x3C) != 0 )
      {
        if ( (v16 & 0x3C0) != 0 )
          return v16;
        goto LABEL_58;
      }
      goto LABEL_73;
    }
    return v16;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v28 + 8) + 8LL) - 17 > 1 || (unsigned __int8)v16 > 0x15u )
    return v16;
  v38 = (_BYTE *)sub_AD7630(v28, 0);
  if ( v38 && *v38 == 18 )
  {
    v29 = (__int64)(v38 + 24);
    goto LABEL_51;
  }
  LOBYTE(v16) = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 85 )
  {
LABEL_19:
    v16 = *(_QWORD *)(a2 - 32);
    if ( !v16 )
      return v16;
    if ( *(_BYTE *)v16 )
      return v16;
    if ( *(_QWORD *)(v16 + 24) != *(_QWORD *)(a2 + 80) )
      return v16;
    if ( *(_DWORD *)(v16 + 36) != 207 )
      return v16;
    v16 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    if ( a1 != *(__int64 **)(a2 - 32 * v16) )
      return v16;
    v16 = 32 * (1 - v16);
    v17 = *(_QWORD *)(a2 + v16);
    if ( *(_BYTE *)v17 != 17 )
      return v16;
    v18 = *(_DWORD *)(v17 + 32);
    if ( v18 > 0x40 )
    {
      LODWORD(v16) = sub_C444A0(v17 + 24);
      if ( v18 - (unsigned int)v16 > 0x40 )
        return v16;
      v19 = **(_QWORD **)(v17 + 24);
    }
    else
    {
      v19 = *(_QWORD *)(v17 + 24);
    }
    v20 = ~(_WORD)v19 & 0x3FF;
    if ( !a4 )
      v20 = v19;
    v21 = ~v20;
    goto LABEL_54;
  }
  v51 = a1;
  v50 = &v48;
  v52 = &v47;
  v53 = 0;
  if ( (unsigned __int8)v16 <= 0x1Cu )
    return v16;
LABEL_60:
  if ( (_BYTE)v16 == 82 )
  {
    v16 = *(_QWORD *)(a2 - 64);
    if ( *(_BYTE *)v16 == 78 )
    {
      v30 = *(__int64 **)(v16 - 32);
      v31 = *(_QWORD *)(v16 + 8);
      v32 = v30[1];
      v33 = *(unsigned __int8 *)(v31 + 8);
      v34 = *(unsigned __int8 *)(v32 + 8);
      LODWORD(v16) = v34 - 17;
      if ( (unsigned int)(v34 - 17) <= 1 == (unsigned int)(v33 - 17) <= 1 )
      {
        if ( (unsigned int)v16 > 1
          || (LODWORD(v16) = *(_DWORD *)(v32 + 32), *(_DWORD *)(v31 + 32) == (_DWORD)v16)
          && (LOBYTE(v16) = (_BYTE)v34 == 18, ((_BYTE)v33 == 18) == ((_BYTE)v34 == 18)) )
        {
          if ( a1 == v30 )
          {
            LOBYTE(v16) = sub_991580((__int64)&v52, *(_QWORD *)(a2 - 32));
            if ( (_BYTE)v16 )
            {
              if ( v50 )
              {
                v35 = sub_B53900(a2);
                v36 = v50;
                *v50 = v35;
                *((_BYTE *)v36 + 4) = BYTE4(v35);
              }
              LOBYTE(v16) = sub_9893F0(v48, v47, &v50);
              if ( (_BYTE)v16 )
              {
                if ( (_BYTE)v50 == a4 )
                {
                  LODWORD(v16) = *(_DWORD *)a6 & 0x3F;
                  *(_DWORD *)a6 = v16;
LABEL_58:
                  *(_WORD *)(a6 + 4) = 257;
                  return v16;
                }
                *(_DWORD *)a6 &= 0x3C3u;
LABEL_73:
                *(_WORD *)(a6 + 4) = 256;
                LOBYTE(v16) = 0;
              }
            }
          }
        }
      }
    }
  }
  return v16;
}
