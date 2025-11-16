// Function: sub_8E4940
// Address: 0x8e4940
//
__int64 __fastcall sub_8E4940(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // r12
  char v7; // al
  char i; // dl
  unsigned int v9; // r13d
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rbx
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 *v25; // r9
  __int64 j; // rdi
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // r12
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // [rsp+8h] [rbp-38h] BYREF
  __int64 v35; // [rsp+10h] [rbp-30h] BYREF
  __int64 v36; // [rsp+18h] [rbp-28h]

  v5 = a2;
  v6 = a1;
  v7 = *(_BYTE *)(a1 + 140);
  if ( v7 != 12 )
    goto LABEL_5;
  do
  {
    v6 = *(_QWORD *)(v6 + 160);
    v7 = *(_BYTE *)(v6 + 140);
  }
  while ( v7 == 12 );
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(v5 + 140) )
  {
    v5 = *(_QWORD *)(v5 + 160);
LABEL_5:
    ;
  }
  v9 = 0;
  if ( v7 != i || HIDWORD(qword_4F077B4) && *(_DWORD *)(v6 + 136) != *(_DWORD *)(v5 + 136) )
    return v9;
  if ( v7 == 2 )
  {
    if ( (*(_BYTE *)(v6 + 161) & 8) != 0 && (*(_BYTE *)(v5 + 161) & 8) != 0 )
      return *(_BYTE *)(v6 + 160) == *(_BYTE *)(v5 + 160);
    goto LABEL_18;
  }
  if ( (unsigned __int8)(v7 - 9) > 2u || (unsigned __int8)(i - 9) > 2u )
  {
LABEL_18:
    v9 = 1;
    if ( v6 != v5 )
      return (unsigned int)sub_8D97D0(v6, v5, 0x20u, HIDWORD(qword_4F077B4), a5) != 0;
    return v9;
  }
  if ( *(char *)(*(_QWORD *)(*(_QWORD *)v6 + 96LL) + 181LL) >= 0
    || *(char *)(*(_QWORD *)(*(_QWORD *)v5 + 96LL) + 181LL) >= 0 )
  {
    goto LABEL_14;
  }
  v11 = **(_QWORD ***)(v6 + 168);
  v12 = **(_QWORD ***)(v5 + 168);
  if ( v11 )
  {
    while ( v12 && (unsigned int)sub_8E4940(v11[5], v12[5]) )
    {
      v11 = (_QWORD *)*v11;
      v12 = (_QWORD *)*v12;
      if ( !v11 )
        goto LABEL_29;
    }
    goto LABEL_14;
  }
LABEL_29:
  if ( v12 )
  {
LABEL_14:
    v9 = 1;
    if ( v6 != v5 )
    {
      v9 = dword_4F07588;
      if ( dword_4F07588 )
        return (*(_QWORD *)(v5 + 32) == *(_QWORD *)(v6 + 32)) & (unsigned __int8)(*(_QWORD *)(v6 + 32) != 0);
    }
    return v9;
  }
  v34 = *(_QWORD *)(v6 + 160);
  v13 = sub_72FD90(v34, 7);
  v14 = *(_QWORD *)(v5 + 160);
  v34 = v13;
  v15 = sub_72FD90(v14, 7);
  v16 = *(_BYTE *)(v6 + 140);
  if ( (unsigned __int8)(v16 - 9) <= 1u )
  {
    if ( (unsigned __int8)(*(_BYTE *)(v5 + 140) - 9) <= 1u )
    {
      while ( 1 )
      {
        if ( !v34 )
          return v15 == 0;
        if ( !v15 )
          break;
        if ( !(unsigned int)sub_8E4D50(v34, v15) )
          return 0;
        v32 = sub_72FD90(*(_QWORD *)(v34 + 112), 7);
        v33 = *(_QWORD *)(v15 + 112);
        v34 = v32;
        v15 = sub_72FD90(v33, 7);
      }
      return v34 == 0;
    }
    return 0;
  }
  if ( v16 != 11 || *(_BYTE *)(v5 + 140) != 11 )
    return 0;
  while ( v34 )
  {
    if ( !v15 )
      return 0;
    v17 = sub_72FD90(*(_QWORD *)(v34 + 112), 7);
    v18 = *(_QWORD *)(v15 + 112);
    v34 = v17;
    v15 = sub_72FD90(v18, 7);
  }
  if ( v15 )
    return 0;
  v19 = sub_823970(256);
  v20 = (_QWORD *)v19;
  do
  {
    if ( v20 )
      *v20 = 0;
    v20 += 2;
  }
  while ( v20 != (_QWORD *)(v19 + 256) );
  v21 = *(_QWORD *)(v6 + 160);
  v35 = v19;
  v36 = 15;
  v34 = sub_72FD90(v21, 7);
LABEL_43:
  if ( v34 )
  {
    for ( j = *(_QWORD *)(v5 + 160); ; j = *(_QWORD *)(v28 + 112) )
    {
      v27 = sub_72FD90(j, 7);
      v28 = v27;
      if ( !v27 )
        break;
      v29 = (v27 >> 3) & v36;
      v30 = (__int64 *)(v35 + 16LL * v29);
      v31 = *v30;
      if ( v28 == *v30 )
      {
LABEL_60:
        if ( v30[1] )
          continue;
      }
      else
      {
        while ( v31 )
        {
          v29 = v36 & (v29 + 1);
          v30 = (__int64 *)(v35 + 16LL * v29);
          v31 = *v30;
          if ( *v30 == v28 )
            goto LABEL_60;
        }
      }
      if ( (unsigned int)sub_8E4D50(v34, v28) )
      {
        sub_8E46E0((__int64)&v35, v28, &v34, v28 >> 3);
        v34 = sub_72FD90(*(_QWORD *)(v34 + 112), 7);
        goto LABEL_43;
      }
    }
  }
  v9 = v34 == 0;
  sub_823A00(v35, 16LL * (unsigned int)(v36 + 1), v22, v23, v24, v25);
  return v9;
}
