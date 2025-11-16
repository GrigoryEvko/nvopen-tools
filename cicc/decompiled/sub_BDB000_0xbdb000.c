// Function: sub_BDB000
// Address: 0xbdb000
//
unsigned __int64 __fastcall sub_BDB000(_BYTE *a1, __int64 a2)
{
  unsigned __int8 v4; // al
  const char *v5; // r15
  unsigned __int8 v6; // dl
  __int64 v7; // rcx
  const char **v8; // rax
  const char **v9; // rdx
  const char *v10; // r13
  __int64 v11; // rbx
  _BYTE *v12; // rax
  unsigned __int64 result; // rax
  char v14; // dl
  int v15; // eax
  __int64 v16; // r13
  _BYTE *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  _BYTE *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // r13
  _BYTE *v24; // rax
  __int64 v25; // rdx
  _QWORD v26[4]; // [rsp+0h] [rbp-60h] BYREF
  char v27; // [rsp+20h] [rbp-40h]
  char v28; // [rsp+21h] [rbp-3Fh]

  if ( (unsigned __int16)sub_AF18C0(a2) != 21 )
  {
    v19 = *(_QWORD *)a1;
    v28 = 1;
    v26[0] = "invalid tag";
    v27 = 3;
    if ( !v19 )
      goto LABEL_26;
    sub_CA0E80(v26, v19);
    v20 = *(_BYTE **)(v19 + 32);
    if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 24) )
    {
      sub_CB5D20(v19, 10);
    }
    else
    {
      *(_QWORD *)(v19 + 32) = v20 + 1;
      *v20 = 10;
    }
    v21 = *(_QWORD *)a1;
    result = (unsigned __int8)a1[154];
    a1[153] = 1;
    a1[152] |= result;
    if ( v21 )
    {
      sub_A62C00((const char *)a2, v21, (__int64)(a1 + 16), *((_QWORD *)a1 + 1));
      v22 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
        return sub_CB5D20(v22, 10);
      }
      else
      {
        *(_QWORD *)(v22 + 32) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
    return result;
  }
  v4 = *(_BYTE *)(a2 - 16);
  if ( (v4 & 2) != 0 )
  {
    v5 = *(const char **)(*(_QWORD *)(a2 - 32) + 24LL);
    if ( v5 )
      goto LABEL_4;
  }
  else
  {
    v5 = *(const char **)(a2 - 16 - 8LL * ((v4 >> 2) & 0xF) + 24);
    if ( v5 )
    {
LABEL_4:
      if ( *v5 != 5 )
      {
        v23 = *(_QWORD *)a1;
        v28 = 1;
        v26[0] = "invalid composite elements";
        v27 = 3;
        if ( !v23 )
          goto LABEL_26;
        sub_CA0E80(v26, v23);
        v24 = *(_BYTE **)(v23 + 32);
        if ( (unsigned __int64)v24 >= *(_QWORD *)(v23 + 24) )
        {
          sub_CB5D20(v23, 10);
        }
        else
        {
          *(_QWORD *)(v23 + 32) = v24 + 1;
          *v24 = 10;
        }
        v25 = *(_QWORD *)a1;
        result = (unsigned __int8)a1[154];
        a1[153] = 1;
        a1[152] |= result;
        if ( v25 )
        {
          sub_BD9900((__int64 *)a1, (const char *)a2);
          return (unsigned __int64)sub_BD9900((__int64 *)a1, v5);
        }
        return result;
      }
      v6 = *(v5 - 16);
      if ( (v6 & 2) != 0 )
      {
        v8 = (const char **)*((_QWORD *)v5 - 4);
        v7 = *((unsigned int *)v5 - 6);
      }
      else
      {
        v7 = (*((_WORD *)v5 - 8) >> 6) & 0xF;
        v8 = (const char **)&v5[-8 * ((v6 >> 2) & 0xF) - 16];
      }
      v9 = &v8[v7];
      if ( v9 != v8 )
      {
        while ( 1 )
        {
          v10 = *v8;
          if ( *v8 )
          {
            if ( *v10 > 0x24u || ((1LL << *v10) & 0x140000F000LL) == 0 )
              break;
          }
          if ( v9 == ++v8 )
            goto LABEL_18;
        }
        v11 = *(_QWORD *)a1;
        v28 = 1;
        v26[0] = "invalid subroutine type ref";
        v27 = 3;
        if ( v11 )
        {
          sub_CA0E80(v26, v11);
          v12 = *(_BYTE **)(v11 + 32);
          if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
          {
            sub_CB5D20(v11, 10);
          }
          else
          {
            *(_QWORD *)(v11 + 32) = v12 + 1;
            *v12 = 10;
          }
          result = *(_QWORD *)a1;
          v14 = a1[154];
          a1[153] = 1;
          a1[152] |= v14;
          if ( result )
          {
            sub_BD9900((__int64 *)a1, (const char *)a2);
            sub_BD9900((__int64 *)a1, v5);
            return (unsigned __int64)sub_BD9900((__int64 *)a1, v10);
          }
          return result;
        }
LABEL_26:
        result = (unsigned __int8)a1[154];
        a1[153] = 1;
        a1[152] |= result;
        return result;
      }
    }
  }
LABEL_18:
  v15 = *(_DWORD *)(a2 + 20);
  if ( (v15 & 0x6000) != 0x6000 )
  {
    result = v15 & 0xC00000;
    if ( (_DWORD)result != 12582912 )
      return result;
  }
  v16 = *(_QWORD *)a1;
  v28 = 1;
  v26[0] = "invalid reference flags";
  v27 = 3;
  if ( !v16 )
    goto LABEL_26;
  sub_CA0E80(v26, v16);
  v17 = *(_BYTE **)(v16 + 32);
  if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 24) )
  {
    sub_CB5D20(v16, 10);
  }
  else
  {
    *(_QWORD *)(v16 + 32) = v17 + 1;
    *v17 = 10;
  }
  v18 = *(_QWORD *)a1;
  result = (unsigned __int8)a1[154];
  a1[153] = 1;
  a1[152] |= result;
  if ( v18 )
    return (unsigned __int64)sub_BD9900((__int64 *)a1, (const char *)a2);
  return result;
}
