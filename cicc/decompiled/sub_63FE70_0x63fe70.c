// Function: sub_63FE70
// Address: 0x63fe70
//
_QWORD *__fastcall sub_63FE70(__int64 a1)
{
  __int64 v1; // r12
  bool v2; // zf
  __int64 *v3; // rbx
  _BOOL4 v5; // r15d
  __int64 *v6; // r14
  __int64 v7; // rbx
  _BOOL4 v8; // r8d
  int v9; // r9d
  char v10; // dl
  _BOOL4 v11; // r14d
  __int64 v12; // r13
  __int64 i; // rdi
  char j; // al
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r13
  _QWORD *v19; // rcx
  _QWORD *v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // r13
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rdx
  char v26; // cl
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-B0h]
  __int64 **v30; // [rsp+18h] [rbp-98h]
  _QWORD *v31; // [rsp+20h] [rbp-90h]
  _QWORD *v32; // [rsp+28h] [rbp-88h]
  __int64 v33; // [rsp+28h] [rbp-88h]
  __int64 v34; // [rsp+28h] [rbp-88h]
  int v35; // [rsp+34h] [rbp-7Ch] BYREF
  __int64 v36; // [rsp+38h] [rbp-78h] BYREF
  _BYTE v37[24]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v38; // [rsp+58h] [rbp-58h]

  v31 = 0;
  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  v2 = *(_BYTE *)(v1 + 140) == 11;
  v30 = *(__int64 ***)(v1 + 168);
  v36 = *(_QWORD *)(a1 + 64);
  if ( !v2 )
  {
    v5 = (*(_BYTE *)(v1 + 176) & 0x20) == 0;
    v6 = *v30;
    if ( *v30 )
    {
      while ( 1 )
      {
        if ( !v5 )
          goto LABEL_12;
        if ( (v6[12] & 2) != 0 )
          goto LABEL_30;
        while ( 1 )
        {
          v6 = (__int64 *)*v6;
          if ( v6 )
            break;
          if ( !v5 )
            goto LABEL_14;
          v5 = 0;
          v6 = *v30;
          if ( !*v30 )
            goto LABEL_14;
LABEL_12:
          while ( (v6[12] & 3) != 1 )
          {
            v6 = (__int64 *)*v6;
            if ( !v6 )
              goto LABEL_14;
          }
LABEL_30:
          v33 = sub_87CF10(v6[5], v1, &v36);
          if ( v33 )
          {
            v21 = sub_726BB0(((*((_BYTE *)v6 + 96) >> 1) ^ 1) & 1);
            *(_BYTE *)(v21 + 9) |= 1u;
            v22 = (_QWORD *)v21;
            *(_QWORD *)(v21 + 16) = v6;
            v23 = sub_725A70(0);
            *(_BYTE *)(v23 + 49) |= 0x20u;
            *(_QWORD *)(v23 + 16) = v33;
            *(_BYTE *)(v33 + 193) |= 0x40u;
            if ( dword_4D048B8 )
            {
              v34 = v23;
              sub_7340D0(v23, 0, 1);
              v23 = v34;
            }
            v22[3] = v23;
            v24 = v31;
            v31 = v22;
            *v22 = v24;
          }
        }
      }
    }
LABEL_14:
    v7 = **(_QWORD **)(*(_QWORD *)v1 + 96LL);
    if ( !v7 )
      goto LABEL_2;
    v8 = 0;
    v9 = 0;
    while ( 1 )
    {
      if ( *(_BYTE *)(v7 + 80) != 8 )
        goto LABEL_16;
      v10 = *(_BYTE *)(*(_QWORD *)(v7 + 104) + 28LL) & 4;
      v11 = (*(_BYTE *)(*(_QWORD *)(v7 + 104) + 28LL) & 8) != 0;
      if ( v8 )
        break;
      v8 = (*(_BYTE *)(*(_QWORD *)(v7 + 104) + 28LL) & 8) != 0;
      if ( v10 )
      {
        v9 = 1;
        goto LABEL_16;
      }
      if ( !v9 )
        goto LABEL_21;
LABEL_16:
      v7 = *(_QWORD *)(v7 + 16);
      if ( !v7 )
        goto LABEL_2;
    }
    if ( v10 )
    {
      v9 = v8;
      v8 = (*(_BYTE *)(*(_QWORD *)(v7 + 104) + 28LL) & 8) != 0;
      goto LABEL_16;
    }
LABEL_21:
    v12 = *(_QWORD *)(v7 + 88);
    for ( i = *(_QWORD *)(v12 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( !(unsigned int)sub_8D3410(i) )
    {
      j = *(_BYTE *)(i + 140);
LABEL_25:
      v8 = v11;
      v9 = 0;
      if ( (unsigned __int8)(j - 9) <= 2u )
      {
        v15 = sub_87CF10(i, i, &v36);
        v9 = 0;
        v8 = v11;
        v28 = v15;
        if ( v15 )
        {
          v16 = sub_726BB0(2);
          *(_BYTE *)(v16 + 9) |= 1u;
          *(_QWORD *)(v16 + 16) = v12;
          v32 = (_QWORD *)v16;
          v17 = sub_725A70(0);
          *(_BYTE *)(v17 + 49) |= 0x20u;
          v18 = v17;
          *(_QWORD *)(v17 + 16) = v28;
          *(_BYTE *)(v28 + 193) |= 0x40u;
          v19 = v32;
          if ( dword_4D048B8 )
          {
            sub_7340D0(v17, 0, 1);
            v19 = v32;
          }
          v20 = v31;
          v19[3] = v18;
          v8 = v11;
          v9 = 0;
          v31 = v19;
          *v19 = v20;
        }
      }
      goto LABEL_16;
    }
    v9 = 0;
    v8 = v11;
    if ( *(_QWORD *)(i + 128) )
    {
      i = sub_8D40F0(i);
      for ( j = *(_BYTE *)(i + 140); j == 12; j = *(_BYTE *)(i + 140) )
        i = *(_QWORD *)(i + 160);
      goto LABEL_25;
    }
    goto LABEL_16;
  }
LABEL_2:
  if ( (*(_BYTE *)(a1 + 192) & 2) != 0 )
  {
    v25 = sub_5F7F60(v1, &v35);
    if ( v35 )
    {
      sub_686C60(769, &v36, v25, *(_QWORD *)a1);
    }
    else if ( v25 )
    {
      v26 = *(_BYTE *)(v25 + 80);
      v27 = v25;
      if ( v26 == 16 )
      {
        v27 = **(_QWORD **)(v25 + 88);
        v26 = *(_BYTE *)(v27 + 80);
      }
      if ( v26 == 24 )
        v27 = *(_QWORD *)(v27 + 88);
      if ( (*(_BYTE *)(*(_QWORD *)(v27 + 88) + 206LL) & 0x10) != 0 )
      {
        sub_6854C0(1776, &v36, v25);
      }
      else if ( (*(_BYTE *)(v25 + 81) & 0x10) != 0 )
      {
        sub_878710(v25, v37);
        if ( dword_4F077C4 == 2 && v38 && (*(_DWORD *)(v38 + 80) & 0x41000) != 0 )
          sub_8841F0(v37, 0, 0, 0);
      }
    }
    else
    {
      sub_6851C0(1555, &v36);
    }
  }
  sub_5F7FF0(v1);
  v3 = v30[23];
  if ( v3 )
  {
    sub_732AE0(v30[23]);
    *((_BYTE *)v3 + 193) |= 0x40u;
  }
  return v31;
}
