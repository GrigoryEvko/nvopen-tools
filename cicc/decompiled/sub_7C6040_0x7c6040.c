// Function: sub_7C6040
// Address: 0x7c6040
//
__int64 __fastcall sub_7C6040(unsigned __int64 a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  int v7; // ebx
  bool v8; // r14
  unsigned __int64 v9; // rsi
  bool v10; // r14
  __int16 v11; // bx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  int v15; // r15d
  unsigned __int16 v16; // ax
  int v17; // edx
  int v18; // r12d
  __int64 v19; // rdx
  __int64 *v21; // rdx
  unsigned int v22; // [rsp+4h] [rbp-4Ch]
  unsigned __int64 v23; // [rsp+8h] [rbp-48h]
  unsigned int v24; // [rsp+10h] [rbp-40h]
  char v25; // [rsp+17h] [rbp-39h]
  int v26; // [rsp+18h] [rbp-38h]
  int v27; // [rsp+1Ch] [rbp-34h]

  v6 = a2 & 4;
  v7 = a2 & 1;
  v8 = !(a2 & 1);
  v23 = a1;
  v9 = a2 & 2;
  v10 = a1 != 0 && v8;
  v24 = v9;
  if ( word_4F06418[0] != 43 )
  {
    if ( word_4F06418[0] == 27 )
    {
      v11 = 28;
    }
    else
    {
      if ( word_4F06418[0] > 0x1Bu )
      {
        if ( word_4F06418[0] == 73 )
        {
          v11 = 74;
          goto LABEL_6;
        }
LABEL_80:
        sub_721090();
      }
      if ( word_4F06418[0] != 25 )
        goto LABEL_80;
      v11 = 26;
    }
LABEL_6:
    v25 = 0;
    if ( v10 )
      sub_7AE360(a1);
    goto LABEL_8;
  }
  if ( v10 )
    sub_7AE360(a1);
  if ( !v7 )
  {
    v25 = 1;
    v11 = 44;
LABEL_8:
    sub_7B8B50(a1, (unsigned int *)v9, a3, a4, a5, a6);
    v15 = 0;
LABEL_9:
    v16 = word_4F06418[0];
    if ( ((unsigned __int8)v25 & (word_4F06418[0] == 42)) != 0 )
    {
      v25 &= word_4F06418[0] == 42;
      v16 = 42;
      a1 = dword_4F07770;
      if ( dword_4F07770 )
      {
        sub_7BC010(dword_4F07770);
        v16 = word_4F06418[0];
      }
    }
    goto LABEL_10;
  }
  sub_7B8B50(a1, (unsigned int *)v9, a3, a4, a5, a6);
  if ( dword_4F077C4 != 2 )
  {
    v25 = 1;
    v15 = 1;
    v11 = 44;
    goto LABEL_9;
  }
  v16 = word_4F06418[0];
  if ( word_4F06418[0] != 1 || (v21 = &qword_4D04A00, (word_4D04A10 & 0x200) == 0) )
  {
    v9 = 0;
    v15 = 1;
    v11 = 44;
    a1 = (v6 == 0 ? 2048 : 0x4000) | 1u;
    sub_7C0F00(a1, 0, (__int64)v21, v12, v13, v14);
    v25 = 1;
    goto LABEL_9;
  }
  v25 = 1;
  v15 = 1;
  v11 = 44;
LABEL_10:
  v27 = 0;
  v17 = -(v6 == 0);
  v26 = 0;
  v18 = 0;
  v19 = ((v17 & 0xFFFFC800) + 0x4000) | 1;
  v22 = v19;
LABEL_11:
  while ( v11 == v16 )
  {
    v19 = v18 | v27 | (unsigned int)v26;
    if ( !(v18 | v27 | v26) )
      return 0;
    if ( v16 == 74 )
      goto LABEL_40;
    while ( 1 )
    {
      while ( 1 )
      {
LABEL_25:
        if ( v11 == 74 )
        {
          if ( v16 != 73 )
          {
            if ( v16 != 74 )
              goto LABEL_51;
            if ( !v18 )
            {
LABEL_44:
              v18 = 0;
              goto LABEL_17;
            }
            goto LABEL_46;
          }
          goto LABEL_56;
        }
        v12 = v24;
        if ( v24 )
        {
          if ( v16 == 75 )
            return 1;
          if ( v16 == 74 )
            goto LABEL_45;
        }
        if ( v16 == 28 )
        {
          v26 = (v26 == 0) + v26 - 1;
          goto LABEL_17;
        }
        if ( v16 > 0x1Cu )
        {
          if ( v16 != 73 )
          {
            if ( v16 == 74 )
              goto LABEL_43;
LABEL_17:
            if ( !v10 )
              goto LABEL_18;
            goto LABEL_34;
          }
LABEL_56:
          ++v18;
          goto LABEL_17;
        }
        if ( v16 == 26 )
        {
          v27 = (v27 == 0) + v27 - 1;
          goto LABEL_17;
        }
        if ( v16 != 27 )
        {
          if ( v16 == 25 )
            goto LABEL_16;
          goto LABEL_51;
        }
        ++v26;
        if ( !v10 )
        {
LABEL_18:
          if ( !v15 )
            goto LABEL_35;
          goto LABEL_19;
        }
LABEL_34:
        a1 = v23;
        sub_7AE360(v23);
        if ( !v15 )
        {
LABEL_35:
          sub_7B8B50(a1, (unsigned int *)v9, v19, v12, v13, v14);
          break;
        }
LABEL_19:
        sub_7B8B50(a1, (unsigned int *)v9, v19, v12, v13, v14);
        if ( dword_4F077C4 != 2 )
          break;
        v16 = word_4F06418[0];
        if ( word_4F06418[0] == 1 )
        {
          v19 = (__int64)&qword_4D04A00;
          if ( (word_4D04A10 & 0x200) != 0 )
            continue;
        }
        a1 = v22;
        v9 = 0;
        sub_7C0F00(v22, 0, v19, v12, v13, v14);
        break;
      }
      v16 = word_4F06418[0];
      if ( word_4F06418[0] != 42 || !v25 )
        goto LABEL_11;
      v19 = dword_4F07770;
      if ( dword_4F07770 )
      {
        if ( !(v18 | v26 | v27) )
          break;
      }
      v16 = 42;
    }
    sub_7BC010(a1);
    v18 = 0;
    v16 = word_4F06418[0];
    v27 = 0;
    v26 = 0;
  }
  if ( v16 != 74 )
  {
    if ( v16 != 25 )
      goto LABEL_25;
    v9 = (unsigned int)dword_4D043F8;
    if ( !dword_4D043F8 )
    {
      if ( v11 == 74 )
        goto LABEL_17;
LABEL_16:
      ++v27;
      goto LABEL_17;
    }
    v9 = 0;
    a1 = 0;
    if ( (unsigned __int16)sub_7BE840(0, 0) != 25 )
    {
      v16 = word_4F06418[0];
      goto LABEL_25;
    }
    v9 = v10;
    a1 = v23;
    sub_7BBD70(v23, (unsigned int *)v10, v19, v12, v13, v14);
    v16 = word_4F06418[0];
LABEL_51:
    if ( v16 == 9 )
      return 0;
    goto LABEL_17;
  }
LABEL_40:
  if ( !v18 )
    return 1;
  if ( v11 == 74 )
    goto LABEL_46;
  if ( !v24 )
  {
LABEL_43:
    if ( !v18 )
      goto LABEL_44;
    goto LABEL_46;
  }
LABEL_45:
  if ( v18 )
  {
LABEL_46:
    --v18;
    goto LABEL_17;
  }
  return 1;
}
