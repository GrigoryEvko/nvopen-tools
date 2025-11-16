// Function: sub_13CF9C0
// Address: 0x13cf9c0
//
__int64 __fastcall sub_13CF9C0(int a1, __int64 **a2, __int64 a3, _QWORD *a4)
{
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 result; // rax
  unsigned int v12; // r15d
  bool v13; // al
  __int64 v14; // rax
  unsigned int v15; // ebx
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned int v18; // edx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  unsigned int v25; // r15d
  __int64 v26; // rax
  char v27; // cl
  unsigned int v28; // ecx
  __int64 v29; // [rsp-60h] [rbp-60h]
  __int64 v30; // [rsp-60h] [rbp-60h]
  int v31; // [rsp-60h] [rbp-60h]
  __int64 v32; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v33; // [rsp-50h] [rbp-50h]
  __int64 v34; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v35; // [rsp-40h] [rbp-40h]

  if ( *(_BYTE *)(a3 + 16) > 0x10u )
    return 0;
  if ( (unsigned __int8)sub_1593BB0(a3) )
    goto LABEL_3;
  if ( *(_BYTE *)(a3 + 16) == 13 )
  {
    v12 = *(_DWORD *)(a3 + 32);
    if ( v12 <= 0x40 )
      v13 = *(_QWORD *)(a3 + 24) == 0;
    else
      v13 = v12 == (unsigned int)sub_16A57B0(a3 + 24);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) != 16 )
      return 0;
    v14 = sub_15A1020(a3);
    if ( !v14 || *(_BYTE *)(v14 + 16) != 13 )
    {
      v31 = *(_QWORD *)(*(_QWORD *)a3 + 32LL);
      if ( v31 )
      {
        v25 = 0;
        while ( 1 )
        {
          v26 = sub_15A0A60(a3, v25);
          if ( !v26 )
            return 0;
          v27 = *(_BYTE *)(v26 + 16);
          if ( v27 != 9 )
          {
            if ( v27 != 13 )
              return 0;
            v28 = *(_DWORD *)(v26 + 32);
            if ( v28 <= 0x40 )
            {
              if ( *(_QWORD *)(v26 + 24) )
                return 0;
            }
            else if ( v28 != (unsigned int)sub_16A57B0(v26 + 24) )
            {
              return 0;
            }
          }
          if ( v31 == ++v25 )
            goto LABEL_3;
        }
      }
      goto LABEL_3;
    }
    v15 = *(_DWORD *)(v14 + 32);
    if ( v15 <= 0x40 )
      v13 = *(_QWORD *)(v14 + 24) == 0;
    else
      v13 = v15 == (unsigned int)sub_16A57B0(v14 + 24);
  }
  if ( !v13 )
    return 0;
LABEL_3:
  v7 = **a2;
  if ( *((_BYTE *)*a2 + 8) == 16 )
  {
    v8 = (*a2)[4];
    v9 = sub_1643320(v7);
    v10 = sub_16463B0(v9, (unsigned int)v8);
  }
  else
  {
    v10 = sub_1643320(v7);
  }
  switch ( a1 )
  {
    case ' ':
    case '%':
      if ( !(unsigned __int8)sub_14BFF20(a2, *a4, 0, a4[3], a4[4], a4[2]) )
        return 0;
      return sub_15A0640(v10);
    case '!':
    case '"':
      if ( !(unsigned __int8)sub_14BFF20(a2, *a4, 0, a4[3], a4[4], a4[2]) )
        return 0;
      return sub_15A0600(v10);
    case '#':
      return sub_15A0600(v10);
    case '$':
      return sub_15A0640(v10);
    case '&':
      sub_14C2530((unsigned int)&v32, (_DWORD)a2, *a4, 0, a4[3], a4[4], a4[2], 0);
      v18 = v35;
      if ( v35 <= 0x40 )
        v19 = v34;
      else
        v19 = *(_QWORD *)(v34 + 8LL * ((v35 - 1) >> 6));
      if ( (v19 & (1LL << ((unsigned __int8)v35 - 1))) != 0 )
        goto LABEL_52;
      v20 = v32;
      if ( v33 > 0x40 )
        v20 = *(_QWORD *)(v32 + 8LL * ((v33 - 1) >> 6));
      if ( (v20 & (1LL << ((unsigned __int8)v33 - 1))) == 0 )
        goto LABEL_64;
      if ( !(unsigned __int8)sub_14BFF20(a2, *a4, 0, a4[3], a4[4], a4[2]) )
        goto LABEL_63;
LABEL_37:
      result = sub_15A0600(v10);
      goto LABEL_53;
    case '\'':
      sub_14C2530((unsigned int)&v32, (_DWORD)a2, *a4, 0, a4[3], a4[4], a4[2], 0);
      if ( v35 <= 0x40 )
        v16 = v34;
      else
        v16 = *(_QWORD *)(v34 + 8LL * ((v35 - 1) >> 6));
      if ( (v16 & (1LL << ((unsigned __int8)v35 - 1))) != 0 )
        goto LABEL_52;
      v17 = v32;
      if ( v33 > 0x40 )
        v17 = *(_QWORD *)(v32 + 8LL * ((v33 - 1) >> 6));
      if ( (v17 & (1LL << ((unsigned __int8)v33 - 1))) == 0 )
        goto LABEL_44;
      goto LABEL_37;
    case '(':
      sub_14C2530((unsigned int)&v32, (_DWORD)a2, *a4, 0, a4[3], a4[4], a4[2], 0);
      if ( v35 <= 0x40 )
        v21 = v34;
      else
        v21 = *(_QWORD *)(v34 + 8LL * ((v35 - 1) >> 6));
      if ( (v21 & (1LL << ((unsigned __int8)v35 - 1))) != 0 )
        goto LABEL_37;
      v22 = v32;
      if ( v33 > 0x40 )
        v22 = *(_QWORD *)(v32 + 8LL * ((v33 - 1) >> 6));
      if ( (v22 & (1LL << ((unsigned __int8)v33 - 1))) != 0 )
        goto LABEL_52;
LABEL_44:
      sub_135E100(&v34);
      sub_135E100(&v32);
      return 0;
    case ')':
      sub_14C2530((unsigned int)&v32, (_DWORD)a2, *a4, 0, a4[3], a4[4], a4[2], 0);
      v18 = v35;
      if ( v35 <= 0x40 )
        v23 = v34;
      else
        v23 = *(_QWORD *)(v34 + 8LL * ((v35 - 1) >> 6));
      if ( (v23 & (1LL << ((unsigned __int8)v35 - 1))) != 0 )
        goto LABEL_37;
      v24 = v32;
      if ( v33 > 0x40 )
        v24 = *(_QWORD *)(v32 + 8LL * ((v33 - 1) >> 6));
      if ( (v24 & (1LL << ((unsigned __int8)v33 - 1))) == 0 )
        goto LABEL_64;
      if ( !(unsigned __int8)sub_14BFF20(a2, *a4, 0, a4[3], a4[4], a4[2]) )
      {
LABEL_63:
        v18 = v35;
LABEL_64:
        if ( v18 > 0x40 && v34 )
          j_j___libc_free_0_0(v34);
        if ( v33 > 0x40 && v32 )
          j_j___libc_free_0_0(v32);
        return 0;
      }
LABEL_52:
      result = sub_15A0640(v10);
LABEL_53:
      if ( v35 > 0x40 && v34 )
      {
        v29 = result;
        j_j___libc_free_0_0(v34);
        result = v29;
      }
      if ( v33 > 0x40 )
      {
        if ( v32 )
        {
          v30 = result;
          j_j___libc_free_0_0(v32);
          result = v30;
        }
      }
      break;
  }
  return result;
}
