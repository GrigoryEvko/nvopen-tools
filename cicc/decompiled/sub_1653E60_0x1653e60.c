// Function: sub_1653E60
// Address: 0x1653e60
//
__int64 __fastcall sub_1653E60(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // r14
  unsigned __int64 v5; // rax
  const char *v6; // rax
  __int64 v7; // rdx
  int v8; // ecx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r15
  unsigned __int8 v12; // dl
  __int64 v13; // r14
  _BYTE *v14; // rax
  __int64 result; // rax
  char v16; // dl
  unsigned __int8 v17; // dl
  const char *v18; // rax
  unsigned __int8 *v19; // rbx
  unsigned __int8 v20; // dl
  int v21; // edx
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rdx
  _QWORD v26[2]; // [rsp+0h] [rbp-50h] BYREF
  char v27; // [rsp+10h] [rbp-40h]
  char v28; // [rsp+11h] [rbp-3Fh]

  v2 = (unsigned __int8 *)a2;
  sub_164F0A0(a1, a2);
  v5 = *(unsigned __int16 *)(a2 + 2);
  if ( (unsigned __int16)v5 > 0x33u || (v7 = 0x8000000880016LL, v8 = (unsigned __int16)v5, !_bittest64(&v7, v5)) )
  {
    v28 = 1;
    v6 = "invalid tag";
    goto LABEL_39;
  }
  v9 = *(unsigned int *)(a2 + 8);
  v10 = v9;
  v11 = *(_QWORD *)(a2 + 8 * (1 - v9));
  if ( v11 )
  {
    v12 = *(_BYTE *)v11;
    if ( *(_BYTE *)v11 > 0x15u )
    {
      if ( (unsigned __int8)(v12 - 31) > 2u )
      {
LABEL_7:
        v13 = *(_QWORD *)a1;
        v28 = 1;
        v26[0] = "invalid scope";
        v27 = 3;
        if ( !v13 )
        {
          result = *(unsigned __int8 *)(a1 + 74);
          *(_BYTE *)(a1 + 73) = 1;
          *(_BYTE *)(a1 + 72) |= result;
          return result;
        }
        sub_16E2CE0(v26, v13);
        v14 = *(_BYTE **)(v13 + 24);
        if ( (unsigned __int64)v14 >= *(_QWORD *)(v13 + 16) )
        {
          sub_16E7DE0(v13, 10);
        }
        else
        {
          *(_QWORD *)(v13 + 24) = v14 + 1;
          *v14 = 10;
        }
        result = *(_QWORD *)a1;
        v16 = *(_BYTE *)(a1 + 74);
        *(_BYTE *)(a1 + 73) = 1;
        *(_BYTE *)(a1 + 72) |= v16;
        if ( result )
          goto LABEL_11;
        return result;
      }
    }
    else if ( v12 <= 0xAu )
    {
      goto LABEL_7;
    }
  }
  v11 = *(_QWORD *)(a2 + 8 * (3 - v9));
  if ( v11 )
  {
    v17 = *(_BYTE *)v11;
    if ( *(_BYTE *)v11 > 0xEu )
    {
      if ( (unsigned __int8)(v17 - 32) > 1u )
        goto LABEL_17;
    }
    else if ( v17 <= 0xAu )
    {
LABEL_17:
      v28 = 1;
      v18 = "invalid base type";
      goto LABEL_18;
    }
  }
  v11 = *(_QWORD *)(a2 + 8 * (4 - v9));
  if ( !v11 || *(_BYTE *)v11 == 4 )
  {
    v19 = *(unsigned __int8 **)(a2 + 8 * (5 - v9));
    if ( v19 )
    {
      v20 = *v19;
      if ( *v19 > 0xEu )
      {
        if ( (unsigned __int8)(v20 - 32) > 1u )
        {
LABEL_26:
          v28 = 1;
          v26[0] = "invalid vtable holder";
          v27 = 3;
          result = sub_16521E0((__int64 *)a1, (__int64)v26);
          if ( *(_QWORD *)a1 )
          {
            sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
            return (__int64)sub_164ED40((__int64 *)a1, v19);
          }
          return result;
        }
      }
      else if ( v20 <= 0xAu )
      {
        goto LABEL_26;
      }
    }
    v21 = *(_DWORD *)(a2 + 28);
    if ( (v21 & 0x6000) == 0x6000 || (v21 & 0xC00000) == 0xC00000 )
    {
      v28 = 1;
      v6 = "invalid reference flags";
      goto LABEL_39;
    }
    if ( (v21 & 0x800) != 0 && (!v11 || *(_DWORD *)(v11 + 8) != 1 || *(_WORD *)(*(_QWORD *)(v11 - 8) + 2LL) != 33) )
    {
      v28 = 1;
      v6 = "invalid vector, expected one element of type subrange";
      goto LABEL_39;
    }
    v22 = *(_QWORD *)(a2 + 8 * (6 - v9));
    if ( v22 )
    {
      sub_16524E0(a1, (unsigned __int8 *)a2, v22);
      v9 = *(unsigned int *)(a2 + 8);
      v8 = *(unsigned __int16 *)(a2 + 2);
      v10 = v9;
    }
    if ( v8 != 23 && v8 != 2 )
    {
LABEL_45:
      result = *(_QWORD *)(a2 + 8 * (8 - v10));
      if ( result && (*(_BYTE *)result != 12 || *(_WORD *)(a2 + 2) != 51) )
      {
        v28 = 1;
        v26[0] = "discriminator can only appear on variant part";
        v27 = 3;
        return sub_16521E0((__int64 *)a1, (__int64)v26);
      }
      return result;
    }
    if ( *(_BYTE *)a2 == 15 )
    {
      v24 = *(_QWORD *)(a2 - 8 * v9);
      if ( !v24 )
        goto LABEL_56;
    }
    else
    {
      v23 = *(_QWORD *)(a2 - 8 * v9);
      if ( !v23 )
      {
LABEL_61:
        v28 = 1;
        v6 = "class/union requires a filename";
LABEL_39:
        v26[0] = v6;
        v27 = 3;
        result = sub_16521E0((__int64 *)a1, (__int64)v26);
        if ( *(_QWORD *)a1 )
          return (__int64)sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
        return result;
      }
      v24 = *(_QWORD *)(v23 - 8LL * *(unsigned int *)(v23 + 8));
      if ( !v24 )
        goto LABEL_60;
    }
    sub_161E970(v24);
    if ( v25 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      goto LABEL_45;
    }
    if ( *(_BYTE *)a2 == 15 )
      goto LABEL_56;
    v9 = *(unsigned int *)(a2 + 8);
LABEL_60:
    v2 = *(unsigned __int8 **)(a2 - 8 * v9);
    if ( !v2 )
      goto LABEL_61;
LABEL_56:
    v28 = 1;
    v26[0] = "class/union requires a filename";
    v27 = 3;
    result = sub_16521E0((__int64 *)a1, (__int64)v26);
    if ( *(_QWORD *)a1 )
    {
      sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
      return (__int64)sub_164ED40((__int64 *)a1, v2);
    }
    return result;
  }
  v28 = 1;
  v18 = "invalid composite elements";
LABEL_18:
  v26[0] = v18;
  v27 = 3;
  result = sub_16521E0((__int64 *)a1, (__int64)v26);
  if ( *(_QWORD *)a1 )
  {
LABEL_11:
    sub_164ED40((__int64 *)a1, (unsigned __int8 *)a2);
    return (__int64)sub_164ED40((__int64 *)a1, (unsigned __int8 *)v11);
  }
  return result;
}
