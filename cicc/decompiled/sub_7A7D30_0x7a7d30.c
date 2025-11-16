// Function: sub_7A7D30
// Address: 0x7a7d30
//
__int64 __fastcall sub_7A7D30(__int64 a1, int a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // r12
  unsigned int v6; // eax
  int v7; // edx
  char v8; // al
  unsigned int v9; // r8d
  int v10; // r14d
  __int64 v11; // rdx
  __int64 v13; // rdi
  unsigned int v14; // eax
  unsigned int v15; // eax
  __int64 v16; // r15
  char v17; // al
  __int64 v18; // rax
  __int64 *v19; // rax
  unsigned int v20; // eax
  unsigned int v21; // [rsp+8h] [rbp-38h] BYREF
  _DWORD v22[13]; // [rsp+Ch] [rbp-34h] BYREF

  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_QWORD *)(a1 + 120);
  v5 = *(_QWORD *)(v3 + 32);
  if ( a2 && HIDWORD(qword_4F077B4) && (v7 = qword_4F077B4) == 0 && qword_4F077A8 <= 0x76BFu )
  {
    if ( *(char *)(v4 + 142) >= 0 && *(_BYTE *)(v4 + 140) == 12 )
    {
      v14 = sub_8D4AB0(v4, HIDWORD(qword_4F077B4), (unsigned int)qword_4F077B4);
      v7 = qword_4F077B4;
    }
    else
    {
      v14 = *(_DWORD *)(v4 + 136);
    }
    v21 = v14;
  }
  else
  {
    v6 = sub_88CF10(v4);
    v7 = qword_4F077B4;
    v21 = v6;
  }
  while ( *(_BYTE *)(v5 + 140) == 12 )
    v5 = *(_QWORD *)(v5 + 160);
  v8 = *(_BYTE *)(a1 + 144) & 1;
  if ( !dword_4F077BC )
  {
    if ( !v7 )
      goto LABEL_16;
LABEL_33:
    if ( dword_4F077C4 != 2 || !qword_4F077A0 )
      goto LABEL_16;
    goto LABEL_35;
  }
  if ( v7 )
    goto LABEL_33;
  if ( qword_4F077A8 <= 0x76BFu )
  {
    if ( a2 )
    {
      v9 = 1;
      if ( v8 )
        return v9;
    }
    goto LABEL_11;
  }
LABEL_35:
  if ( !v8 )
  {
    v15 = *(_DWORD *)(v5 + 184);
    if ( !v15 || v15 >= v21 )
      goto LABEL_11;
    if ( sub_736C60(57, *(__int64 **)(v5 + 104)) )
    {
      v16 = *(_QWORD *)(a1 + 120);
      if ( (unsigned int)sub_8D3410(v16) )
        v16 = sub_8D40F0(v16);
      while ( 1 )
      {
        v17 = *(_BYTE *)(v16 + 140);
        if ( v17 != 12 )
          break;
        v16 = *(_QWORD *)(v16 + 160);
      }
      if ( (unsigned __int8)(v17 - 9) <= 2u && (*(_BYTE *)(v16 + 179) & 0x20) == 0 && *(_QWORD *)v16 )
      {
        v18 = *(_QWORD *)(*(_QWORD *)v16 + 96LL);
        if ( (_DWORD)qword_4F077B4 )
        {
          if ( *(char *)(v18 + 178) >= 0 )
          {
LABEL_48:
            v19 = sub_736C60(85, *(__int64 **)(v5 + 104));
            if ( v19 )
            {
              v20 = sub_620FD0(*(_QWORD *)(v19[4] + 40), v22);
              if ( v21 > v20 )
                v21 = v20;
            }
            if ( !a2 )
            {
              v10 = 1;
              sub_685330(0x8D6u, (_DWORD *)(a1 + 64), *(_QWORD *)(a1 + 120));
              goto LABEL_12;
            }
            v10 = 1;
            if ( (*(_BYTE *)(a1 + 144) & 1) == 0 )
              goto LABEL_12;
LABEL_19:
            v9 = 1;
            if ( qword_4F077A8 <= 0x76BFu )
              return v9;
            goto LABEL_12;
          }
        }
        else if ( *(char *)(v18 + 181) >= 0 || (*(_BYTE *)(v18 + 178) & 0x40) == 0 )
        {
          goto LABEL_48;
        }
      }
    }
    v8 = *(_BYTE *)(a1 + 144) & 1;
  }
LABEL_16:
  if ( a2 && v8 )
  {
    v10 = 0;
    goto LABEL_19;
  }
LABEL_11:
  v10 = 0;
LABEL_12:
  if ( (unsigned int)sub_7A6510(a1, &v21) )
    return v21;
  if ( !dword_4D0425C || (*(_BYTE *)(a1 + 144) & 4) == 0 || *(_BYTE *)(a1 + 137) || *(_BYTE *)(v5 + 140) == 11 )
  {
    if ( !v10 )
      sub_7A65D0(&v21, v5);
    return v21;
  }
  v13 = *(_QWORD *)(a1 + 120);
  if ( *(char *)(v13 + 142) >= 0 && *(_BYTE *)(v13 + 140) == 12 )
    return (unsigned int)sub_8D4AB0(v13, &v21, v11);
  else
    return *(unsigned int *)(v13 + 136);
}
