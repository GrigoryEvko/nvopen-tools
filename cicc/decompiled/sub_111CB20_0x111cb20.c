// Function: sub_111CB20
// Address: 0x111cb20
//
_QWORD *__fastcall sub_111CB20(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  _QWORD *v6; // r15
  __int64 v7; // r13
  __int16 v8; // r14
  _QWORD **v9; // rdx
  int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // rsi
  char *v14; // rbx
  char v15; // al
  unsigned __int8 v16; // dl
  __int64 v17; // r9
  __int64 v18; // r13
  __int64 v19; // r15
  __int64 v20; // rbx
  unsigned __int8 *v21; // rdi
  __int64 v22; // r9
  unsigned __int8 v23; // al
  __int64 v24; // r13
  __int16 v25; // r12
  __int16 v26; // r12
  __int64 v27; // rcx
  __int64 v28; // rsi
  int v29; // eax
  unsigned __int64 v30; // rdi
  unsigned __int8 v31; // al
  __int64 v32; // r13
  _BYTE *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // [rsp+0h] [rbp-80h]
  __int64 v38; // [rsp+0h] [rbp-80h]
  __int64 v39; // [rsp+8h] [rbp-78h]
  __int64 v40; // [rsp+8h] [rbp-78h]
  __int64 v41; // [rsp+8h] [rbp-78h]
  __int64 v42; // [rsp+18h] [rbp-68h]
  _BYTE v43[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v44; // [rsp+40h] [rbp-40h]

  v4 = sub_F0C930((__int64)a1, *(_QWORD *)(a2 - 64));
  v5 = sub_F0C930((__int64)a1, *(_QWORD *)(a2 - 32));
  v6 = (_QWORD *)(v5 | v4);
  if ( v5 | v4 )
  {
    v7 = v5;
    v8 = *(_WORD *)(a2 + 2) & 0x3F;
    if ( v4 )
    {
      if ( v5 )
        goto LABEL_4;
    }
    else
    {
      v4 = *(_QWORD *)(a2 - 64);
      if ( v5 )
        goto LABEL_4;
    }
    v7 = *(_QWORD *)(a2 - 32);
LABEL_4:
    v44 = 257;
    v6 = sub_BD2C40(72, unk_3F10FD0);
    if ( v6 )
    {
      v9 = *(_QWORD ***)(v4 + 8);
      v10 = *((unsigned __int8 *)v9 + 8);
      if ( (unsigned int)(v10 - 17) > 1 )
      {
        v12 = sub_BCB2A0(*v9);
      }
      else
      {
        BYTE4(v42) = (_BYTE)v10 == 18;
        LODWORD(v42) = *((_DWORD *)v9 + 8);
        v11 = (__int64 *)sub_BCB2A0(*v9);
        v12 = sub_BCE1B0(v11, v42);
      }
      sub_B523C0((__int64)v6, v12, 53, v8, v4, v7, (__int64)v43, 0, 0, 0);
    }
    return v6;
  }
  v14 = *(char **)(a2 - 64);
  v15 = *v14;
  if ( (unsigned __int8)(*v14 - 67) > 0xCu )
    return v6;
  v16 = **(_BYTE **)(a2 - 32);
  if ( v16 > 0x15u && (unsigned __int8)(v16 - 67) > 0xCu )
    return v6;
  v17 = *((_QWORD *)v14 - 4);
  v18 = *((_QWORD *)v14 + 1);
  v19 = *(_QWORD *)(v17 + 8);
  if ( v15 == 76 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 > 1 )
    {
      v37 = *((_QWORD *)v14 + 1);
      v28 = *(_QWORD *)(v17 + 8);
    }
    else
    {
      v28 = *(_QWORD *)(v19 + 24);
      v37 = *(_QWORD *)(v18 + 24);
    }
    v41 = *((_QWORD *)v14 - 4);
    v29 = sub_AE43A0(a1[11], v28);
    v17 = v41;
    if ( v29 != *(_DWORD *)(v37 + 8) >> 8 )
      goto LABEL_35;
    v30 = *(_QWORD *)(a2 - 32);
    v31 = *(_BYTE *)v30;
    if ( *(_BYTE *)v30 > 0x1Cu )
    {
      if ( v31 != 76 )
      {
LABEL_35:
        v15 = *v14;
        goto LABEL_12;
      }
      goto LABEL_40;
    }
    if ( v31 == 5 )
    {
      if ( *(_WORD *)(v30 + 2) == 47 )
      {
LABEL_40:
        v35 = *(_QWORD *)(v30 - 32);
        if ( !v35 || *(_QWORD *)(v35 + 8) != *(_QWORD *)(v41 + 8) )
          goto LABEL_35;
        goto LABEL_42;
      }
    }
    else if ( v31 > 0x15u )
    {
      goto LABEL_35;
    }
    v36 = sub_AD4C70(v30, (__int64 **)v19, 0);
    v17 = v41;
    v35 = v36;
    if ( !v36 )
      goto LABEL_35;
LABEL_42:
    v38 = v35;
    v40 = v17;
    v26 = *(_WORD *)(a2 + 2) & 0x3F;
    v44 = 257;
    v6 = sub_BD2C40(72, unk_3F10FD0);
    if ( !v6 )
      return v6;
    v27 = v38;
    goto LABEL_29;
  }
LABEL_12:
  if ( v15 == 77 )
  {
    v20 = v19;
    if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
    {
      v18 = *(_QWORD *)(v18 + 24);
      v20 = *(_QWORD *)(v19 + 24);
    }
    v39 = v17;
    if ( (unsigned int)sub_AE43A0(a1[11], v18) == *(_DWORD *)(v20 + 8) >> 8 )
    {
      v21 = *(unsigned __int8 **)(a2 - 32);
      v22 = v39;
      v23 = *v21;
      if ( *v21 <= 0x1Cu )
      {
        if ( v23 <= 0x15u )
        {
          v32 = a1[11];
          v33 = (_BYTE *)sub_AD4C50((unsigned __int64)v21, (__int64 **)v19, 0);
          v34 = sub_97B670(v33, v32, 0);
          v22 = v39;
          v24 = v34;
          if ( v34 )
          {
LABEL_27:
            v25 = *(_WORD *)(a2 + 2);
            v40 = v22;
            v44 = 257;
            v26 = v25 & 0x3F;
            v6 = sub_BD2C40(72, unk_3F10FD0);
            if ( v6 )
            {
              v27 = v24;
LABEL_29:
              sub_1113300((__int64)v6, v26, v40, v27, (__int64)v43);
            }
            return v6;
          }
        }
      }
      else if ( v23 == 77 )
      {
        v24 = *((_QWORD *)v21 - 4);
        if ( v24 )
        {
          if ( *(_QWORD *)(v24 + 8) == *(_QWORD *)(v39 + 8) )
            goto LABEL_27;
        }
      }
    }
  }
  v6 = sub_111C5E0(a1, a2);
  if ( v6 )
    return v6;
  return sub_1116C00((__int64)a1, a2);
}
