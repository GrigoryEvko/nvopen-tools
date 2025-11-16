// Function: sub_819840
// Address: 0x819840
//
char *__fastcall sub_819840(_QWORD *a1, const char **a2, _DWORD *a3)
{
  unsigned __int8 *v3; // r12
  char v4; // bl
  unsigned __int8 *v5; // r14
  int v6; // r15d
  const char *v7; // r13
  unsigned __int8 v8; // r12
  int v9; // edx
  const char *v10; // r15
  int v12; // eax
  int v13; // r9d
  _QWORD *v14; // rax
  int v16; // [rsp+14h] [rbp-5Ch]
  unsigned __int8 *v17; // [rsp+18h] [rbp-58h]
  _BYTE v18[5]; // [rsp+23h] [rbp-4Dh]
  unsigned __int8 *v19; // [rsp+28h] [rbp-48h]
  int v20[13]; // [rsp+3Ch] [rbp-34h] BYREF

  if ( a2 )
    *a2 = 0;
  if ( (_DWORD)qword_4F077B4 && qword_4F077A0 <= 0x765Bu )
  {
    v3 = (unsigned __int8 *)a1[8];
    v19 = &v3[a1[7]];
  }
  else
  {
    v3 = (unsigned __int8 *)a1[2];
    v19 = &v3[a1[1]];
  }
  if ( v3 == v19 )
  {
LABEL_56:
    v10 = 0;
    sub_684AA0(7u, 0x9DCu, a3);
    return (char *)v10;
  }
  v16 = 0;
  v4 = 0;
  v17 = 0;
  v5 = v3;
  v6 = 0;
  v7 = 0;
  do
  {
    v8 = v4 ^ 1;
    v9 = (char)*v5;
    if ( !dword_4F055C0[v9 + 128] )
    {
      v18[4] = 0;
      *(_DWORD *)v18 = v7 == 0;
      v12 = sub_7B3CF0(v5, v20, *(int *)v18);
      v13 = *(_DWORD *)&v18[1];
      if ( v12 )
      {
        if ( v18[0] )
        {
          if ( v6 )
            goto LABEL_49;
        }
        else
        {
          if ( !v6 )
          {
LABEL_38:
            v17 = v5;
            v6 = v13;
            v5 += v20[0];
            continue;
          }
LABEL_49:
          if ( v8 )
          {
            v7 = (const char *)v5;
            v8 = 0;
            v4 = 1;
            sub_684AA0(7u, 0x9DCu, a3);
            v13 = *(_DWORD *)&v18[1];
            goto LABEL_38;
          }
        }
        v13 = v6;
        v7 = (const char *)v5;
        goto LABEL_38;
      }
      LOBYTE(v9) = *v5;
    }
    if ( v7 )
      v6 = 1;
    if ( (_BYTE)v9 == 32 )
    {
      ++v5;
      continue;
    }
    if ( (_BYTE)v9 == 10 )
    {
      v14 = (_QWORD *)sub_7AF1D0((unsigned __int64)v5);
      if ( v14[7] != v14[8] )
      {
        v5 = (unsigned __int8 *)v14[7];
        continue;
      }
LABEL_42:
      v5 += v14[4];
      continue;
    }
    if ( (_BYTE)v9 == 58 && a2 )
    {
      if ( v5[1] != 58 )
        goto LABEL_16;
      if ( (v8 & (v7 == 0)) != 0 )
      {
        v8 = 0;
        v7 = 0;
        v4 = 1;
        sub_684AA0(7u, 0xB87u, a3);
      }
      else
      {
        if ( v16 && v8 )
        {
          v8 = 0;
          v4 = 1;
          sub_684AA0(7u, 0xB88u, a3);
        }
        v16 = 1;
        v5 += 2;
        v6 = 0;
        *a2 = v7;
        v7 = 0;
      }
    }
    else
    {
      if ( (_BYTE)v9 )
      {
LABEL_16:
        if ( v8 )
          goto LABEL_56;
        return 0;
      }
      if ( v5[1] != 3 )
      {
        v5 += 2;
        continue;
      }
      v14 = sub_7AEFF0((unsigned __int64)v5);
      v5 = (unsigned __int8 *)v14[2];
      if ( v5 )
        goto LABEL_42;
      v5 = (unsigned __int8 *)qword_4F06498;
      if ( !unk_4F06478 )
        goto LABEL_42;
      v5 = (unsigned __int8 *)(v14[4] + unk_4F06470 + qword_4F06498);
    }
  }
  while ( v5 != v19 );
  v10 = v7;
  if ( v7 )
  {
    if ( !a2
      && (unsigned __int64)(v17 - (unsigned __int8 *)v7 - 4) <= 0x3A
      && *v7 == 95
      && v7[1] == 95
      && *v17 == 95
      && *(v17 - 1) == 95 )
    {
      strncpy(byte_4F19420, v7 + 2, 0x40u);
      byte_4F19420[v17 - (unsigned __int8 *)v7 - 3] = 0;
      return byte_4F19420;
    }
  }
  else if ( v8 )
  {
    goto LABEL_56;
  }
  return (char *)v10;
}
