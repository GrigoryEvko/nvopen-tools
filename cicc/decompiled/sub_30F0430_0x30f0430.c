// Function: sub_30F0430
// Address: 0x30f0430
//
void __fastcall sub_30F0430(__int64 a1, unsigned __int8 *a2)
{
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 v6; // r8
  unsigned __int8 *v7; // r13
  __int64 v8; // rcx
  const char **v9; // r10
  _BYTE *v10; // rax
  _BYTE *v11; // rax
  __int64 v12; // rsi
  bool v13; // al
  unsigned __int64 v14; // rdx
  int v15; // r15d
  unsigned int v16; // r14d
  unsigned __int8 *v17; // rax
  unsigned int v18; // esi
  unsigned int v19; // r8d
  int v20; // eax
  __int64 v21; // r9
  unsigned int v22; // r13d
  int v23; // eax
  bool v24; // r14
  unsigned int v25; // [rsp+8h] [rbp-78h]
  unsigned int v26; // [rsp+Ch] [rbp-74h]
  __int64 v27; // [rsp+10h] [rbp-70h]
  unsigned __int64 v28; // [rsp+10h] [rbp-70h]
  const char **v29; // [rsp+10h] [rbp-70h]
  const char **v30; // [rsp+10h] [rbp-70h]
  __int64 v31; // [rsp+18h] [rbp-68h]
  __int64 v32; // [rsp+18h] [rbp-68h]
  unsigned __int64 v33; // [rsp+18h] [rbp-68h]
  unsigned __int64 v34; // [rsp+18h] [rbp-68h]
  const char **v35; // [rsp+18h] [rbp-68h]
  const char **v36; // [rsp+18h] [rbp-68h]
  const char *v37; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v38; // [rsp+28h] [rbp-58h]
  unsigned __int64 v39; // [rsp+30h] [rbp-50h]
  unsigned int v40; // [rsp+38h] [rbp-48h]
  char v41; // [rsp+40h] [rbp-40h]
  char v42; // [rsp+41h] [rbp-3Fh]

  v3 = *(_QWORD *)(a1 + 32);
  v4 = *(_QWORD *)(a1 + 40);
  v5 = sub_B43CC0((__int64)a2);
  v7 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v8 = *v7;
  if ( (unsigned int)(v8 - 12) <= 1 )
    goto LABEL_2;
  v12 = *((_QWORD *)v7 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 > 1 )
  {
    v21 = 0;
    if ( (unsigned __int8)v8 >= 0x1Du )
      v21 = (__int64)v7;
    sub_9AC3E0((__int64)&v37, (__int64)v7, v5, 0, v3, v21, v4, 1);
    v22 = v38;
    v9 = &v37;
    if ( v38 )
    {
      if ( v38 <= 0x40 )
      {
        v24 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38) == (_QWORD)v37;
        if ( v40 <= 0x40 )
          goto LABEL_36;
      }
      else
      {
        v23 = sub_C445E0((__int64)&v37);
        v9 = &v37;
        v24 = v22 == v23;
        if ( v40 <= 0x40 )
          goto LABEL_34;
      }
    }
    else
    {
      if ( v40 <= 0x40 )
        goto LABEL_3;
      v24 = 1;
    }
    if ( v39 )
    {
      j_j___libc_free_0_0(v39);
      v22 = v38;
      v9 = &v37;
    }
    if ( v22 <= 0x40 )
    {
LABEL_36:
      if ( !v24 )
        return;
LABEL_3:
      v42 = 1;
      v37 = "Undefined behavior: Division by zero";
      v41 = 3;
      sub_CA0E80((__int64)v9, a1 + 88);
      v10 = *(_BYTE **)(a1 + 120);
      if ( (unsigned __int64)v10 >= *(_QWORD *)(a1 + 112) )
      {
        sub_CB5D20(a1 + 88, 10);
      }
      else
      {
        *(_QWORD *)(a1 + 120) = v10 + 1;
        *v10 = 10;
      }
      if ( *a2 <= 0x1Cu )
      {
        sub_A5BF40(a2, a1 + 88, 1, *(_QWORD *)a1);
        v11 = *(_BYTE **)(a1 + 120);
        if ( (unsigned __int64)v11 < *(_QWORD *)(a1 + 112) )
          goto LABEL_7;
      }
      else
      {
        sub_A69870((__int64)a2, (_BYTE *)(a1 + 88), 0);
        v11 = *(_BYTE **)(a1 + 120);
        if ( (unsigned __int64)v11 < *(_QWORD *)(a1 + 112) )
        {
LABEL_7:
          *(_QWORD *)(a1 + 120) = v11 + 1;
          *v11 = 10;
          return;
        }
      }
      sub_CB5D20(a1 + 88, 10);
      return;
    }
LABEL_34:
    if ( v37 )
    {
      j_j___libc_free_0_0((unsigned __int64)v37);
      v9 = &v37;
    }
    goto LABEL_36;
  }
  v27 = *((_QWORD *)v7 + 1);
  v31 = v5;
  if ( (unsigned __int8)v8 <= 0x15u )
  {
    v13 = sub_AD7890((__int64)v7, v12, v5, v8, v6);
    v14 = v31;
    if ( v13 )
    {
LABEL_2:
      v9 = &v37;
      goto LABEL_3;
    }
    v15 = *(_DWORD *)(v27 + 32);
    if ( v15 )
    {
      v16 = 0;
      v9 = &v37;
      while ( 1 )
      {
        v32 = (__int64)v9;
        v28 = v14;
        v17 = (unsigned __int8 *)sub_AD69F0(v7, v16);
        v9 = (const char **)v32;
        if ( (unsigned int)*v17 - 12 <= 1 )
          goto LABEL_3;
        sub_9AC3E0(v32, (__int64)v17, v28, 0, 0, 0, 0, 1);
        v18 = v38;
        v9 = (const char **)v32;
        v19 = v38;
        if ( !v38 )
          break;
        v14 = v28;
        if ( v38 <= 0x40 )
        {
          if ( v37 == (const char *)(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v38)) )
          {
LABEL_45:
            if ( v40 > 0x40 )
              goto LABEL_46;
            goto LABEL_48;
          }
        }
        else
        {
          v25 = v38;
          v26 = v38;
          v20 = sub_C445E0(v32);
          v18 = v25;
          v9 = (const char **)v32;
          v14 = v28;
          v19 = v26;
          if ( v25 == v20 )
            goto LABEL_45;
        }
        if ( v40 > 0x40 && v39 )
        {
          v29 = v9;
          v33 = v14;
          j_j___libc_free_0_0(v39);
          v19 = v38;
          v9 = v29;
          v14 = v33;
        }
        if ( v19 > 0x40 && v37 )
        {
          v30 = v9;
          v34 = v14;
          j_j___libc_free_0_0((unsigned __int64)v37);
          v9 = v30;
          v14 = v34;
        }
        if ( v15 == ++v16 )
          return;
      }
      if ( v40 <= 0x40 )
        goto LABEL_3;
LABEL_46:
      if ( v39 )
      {
        v35 = v9;
        j_j___libc_free_0_0(v39);
        v18 = v38;
        v9 = v35;
      }
LABEL_48:
      if ( v18 > 0x40 && v37 )
      {
        v36 = v9;
        j_j___libc_free_0_0((unsigned __int64)v37);
        v9 = v36;
      }
      goto LABEL_3;
    }
  }
}
