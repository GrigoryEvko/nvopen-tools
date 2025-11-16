// Function: sub_29CF3C0
// Address: 0x29cf3c0
//
__int64 __fastcall sub_29CF3C0(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  unsigned __int8 *v7; // rbx
  unsigned __int8 v8; // al
  int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // rcx
  unsigned __int16 v14; // ax
  __int64 *v15; // rdx
  __int64 *v16; // r15
  unsigned __int8 **v17; // rax
  char v18; // dl
  unsigned __int8 **v19; // rax
  __int64 v20; // r12
  __int64 *v21; // rax
  char v22; // dl
  unsigned __int8 **v23; // rax
  unsigned __int8 **v24; // rax
  __int64 v25; // [rsp+10h] [rbp-40h]
  char v26; // [rsp+18h] [rbp-38h]

  v7 = a1;
  v8 = *a1;
  if ( *a1 <= 3u )
  {
LABEL_2:
    LOBYTE(v6) = (v7[33] & 0x1C) == 0;
    if ( (v7[33] & 3) == 1 )
      return 0;
    return v6;
  }
  v11 = v8;
  while ( 1 )
  {
    v12 = *((_DWORD *)v7 + 1) & 0x7FFFFFF;
    LOBYTE(v6) = (*((_DWORD *)v7 + 1) & 0x7FFFFFF) == 0 || v8 == 4;
    if ( (_BYTE)v6 )
      return 1;
    v13 = (unsigned int)(v11 - 9);
    if ( (unsigned int)v13 <= 2 )
    {
      v15 = (__int64 *)(32LL * (unsigned int)v12);
      if ( (v7[7] & 0x40) != 0 )
      {
        v16 = (__int64 *)*((_QWORD *)v7 - 1);
        v7 = (unsigned __int8 *)v15 + (_QWORD)v16;
      }
      else
      {
        v16 = (__int64 *)(v7 - (unsigned __int8 *)v15);
      }
      while ( 1 )
      {
        v20 = *v16;
        if ( *(_BYTE *)(a2 + 28) )
        {
          v21 = *(__int64 **)(a2 + 8);
          v13 = *(unsigned int *)(a2 + 20);
          v15 = &v21[v13];
          if ( v21 != v15 )
          {
            while ( v20 != *v21 )
            {
              if ( v15 == ++v21 )
                goto LABEL_44;
            }
            goto LABEL_18;
          }
LABEL_44:
          if ( (unsigned int)v13 < *(_DWORD *)(a2 + 16) )
          {
            *(_DWORD *)(a2 + 20) = v13 + 1;
            *v15 = v20;
            ++*(_QWORD *)a2;
LABEL_17:
            v6 = sub_29CF3C0(v20, a2, a3);
            if ( !(_BYTE)v6 )
              return v6;
            goto LABEL_18;
          }
        }
        sub_C8CC70(a2, *v16, (__int64)v15, v13, a5, a6);
        if ( (_BYTE)v15 )
          goto LABEL_17;
LABEL_18:
        v16 += 4;
        if ( v7 == (unsigned __int8 *)v16 )
          return 1;
      }
    }
    v14 = *((_WORD *)v7 + 1);
    if ( v14 > 0x30u )
    {
      if ( v14 != 49 )
        return v6;
      v12 = -32LL * (unsigned int)v12;
      v7 = *(unsigned __int8 **)&v7[v12];
      if ( !*(_BYTE *)(a2 + 28) )
        goto LABEL_27;
      v17 = *(unsigned __int8 ***)(a2 + 8);
      v13 = *(unsigned int *)(a2 + 20);
      v12 = (__int64)&v17[v13];
      if ( v17 != (unsigned __int8 **)v12 )
      {
        while ( v7 != *v17 )
        {
          if ( (unsigned __int8 **)v12 == ++v17 )
            goto LABEL_62;
        }
        return 1;
      }
      goto LABEL_62;
    }
    if ( v14 > 0x2Eu )
    {
      v25 = sub_9208B0(a3, *(_QWORD *)(*(_QWORD *)&v7[-32 * (unsigned int)v12] + 8LL));
      v26 = v22;
      if ( sub_9208B0(a3, *((_QWORD *)v7 + 1)) != v25 || (_BYTE)v12 != v26 )
        return v6;
      v7 = *(unsigned __int8 **)&v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
      if ( !*(_BYTE *)(a2 + 28) )
      {
LABEL_27:
        sub_C8CC70(a2, (__int64)v7, v12, v13, a5, a6);
        if ( !v18 )
          return 1;
        goto LABEL_28;
      }
      v23 = *(unsigned __int8 ***)(a2 + 8);
      v13 = *(unsigned int *)(a2 + 20);
      v12 = (__int64)&v23[v13];
      if ( v23 != (unsigned __int8 **)v12 )
      {
        while ( v7 != *v23 )
        {
          if ( (unsigned __int8 **)v12 == ++v23 )
            goto LABEL_62;
        }
        return 1;
      }
      goto LABEL_62;
    }
    if ( v14 != 13 )
      break;
    if ( **(_BYTE **)&v7[32 * (1LL - (unsigned int)v12)] != 17 )
      return 0;
    v12 = -32LL * (unsigned int)v12;
    v7 = *(unsigned __int8 **)&v7[v12];
    if ( !*(_BYTE *)(a2 + 28) )
      goto LABEL_27;
    v24 = *(unsigned __int8 ***)(a2 + 8);
    v13 = *(unsigned int *)(a2 + 20);
    v12 = (__int64)&v24[v13];
    if ( v24 != (unsigned __int8 **)v12 )
    {
      while ( v7 != *v24 )
      {
        if ( (unsigned __int8 **)v12 == ++v24 )
          goto LABEL_62;
      }
      return 1;
    }
LABEL_62:
    if ( (unsigned int)v13 >= *(_DWORD *)(a2 + 16) )
      goto LABEL_27;
    *(_DWORD *)(a2 + 20) = v13 + 1;
    *(_QWORD *)v12 = v7;
    ++*(_QWORD *)a2;
LABEL_28:
    v11 = *v7;
    v8 = *v7;
    if ( (unsigned __int8)v11 <= 3u )
      goto LABEL_2;
  }
  if ( v14 != 34 )
    return v6;
  if ( (_DWORD)v12 == 1 )
  {
LABEL_30:
    v7 = *(unsigned __int8 **)&v7[-32 * (unsigned int)v12];
    if ( !*(_BYTE *)(a2 + 28) )
      goto LABEL_27;
    v19 = *(unsigned __int8 ***)(a2 + 8);
    v13 = *(unsigned int *)(a2 + 20);
    v12 = (__int64)&v19[v13];
    if ( v19 != (unsigned __int8 **)v12 )
    {
      while ( v7 != *v19 )
      {
        if ( (unsigned __int8 **)v12 == ++v19 )
          goto LABEL_62;
      }
      return 1;
    }
    goto LABEL_62;
  }
  LODWORD(v13) = 1;
  while ( **(_BYTE **)&v7[32 * ((unsigned int)v13 - (unsigned __int64)(unsigned int)v12)] == 17 )
  {
    v13 = (unsigned int)(v13 + 1);
    if ( (_DWORD)v12 == (_DWORD)v13 )
      goto LABEL_30;
  }
  return 0;
}
