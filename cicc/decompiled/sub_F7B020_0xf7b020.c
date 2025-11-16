// Function: sub_F7B020
// Address: 0xf7b020
//
__int64 __fastcall sub_F7B020(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int16 v6; // ax
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rcx
  _BYTE *v11; // r8
  _BYTE *v12; // rsi
  __int64 v13; // rdi
  unsigned __int16 v14; // dx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // r12
  __int64 *v20; // r14
  __int64 v21; // rbx
  __int64 *v22; // rax
  __int16 v23; // ax
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  _BYTE *v27; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v28; // [rsp+20h] [rbp-E0h]
  _BYTE v29[64]; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v30; // [rsp+68h] [rbp-98h] BYREF
  __int64 *v31; // [rsp+70h] [rbp-90h]
  __int64 v32; // [rsp+78h] [rbp-88h]
  int v33; // [rsp+80h] [rbp-80h]
  char v34; // [rsp+84h] [rbp-7Ch]
  __int64 v35; // [rsp+88h] [rbp-78h] BYREF

  v28 = 0x800000000LL;
  v31 = &v35;
  v32 = 0x100000008LL;
  v6 = *(_WORD *)(a1 + 24);
  v27 = v29;
  v33 = 0;
  v34 = 1;
  v35 = a1;
  v30 = 1;
  if ( v6 == 7 )
  {
    if ( !(unsigned __int8)sub_DBE090(*(_QWORD *)a2, *(_QWORD *)(a1 + 40)) )
      goto LABEL_11;
    v6 = *(_WORD *)(a1 + 24);
  }
  if ( v6 != 8 || sub_D4B130(*(_QWORD *)(a1 + 48)) || *(_BYTE *)(a2 + 8) && *(_QWORD *)(a1 + 40) == 2 )
  {
    v7 = (unsigned int)v28;
    v8 = (unsigned int)v28 + 1LL;
    if ( v8 > HIDWORD(v28) )
    {
      sub_C8D5F0((__int64)&v27, v29, v8, 8u, a5, a6);
      v7 = (unsigned int)v28;
    }
    *(_QWORD *)&v27[8 * v7] = a1;
    result = (unsigned int)(v28 + 1);
    LODWORD(v28) = v28 + 1;
    goto LABEL_12;
  }
LABEL_11:
  *(_BYTE *)(a2 + 9) = 1;
  result = (unsigned int)v28;
LABEL_12:
  v10 = a2;
  while ( 1 )
  {
    v11 = v27;
    v12 = &v27[8 * (unsigned int)result];
    if ( !(_DWORD)result )
      break;
    while ( 1 )
    {
      if ( *(_BYTE *)(v10 + 9) )
        goto LABEL_18;
      v13 = *((_QWORD *)v12 - 1);
      result = (unsigned int)(result - 1);
      LODWORD(v28) = result;
      v14 = *(_WORD *)(v13 + 24);
      if ( v14 > 0xEu )
      {
        if ( v14 != 15 )
          BUG();
        goto LABEL_17;
      }
      if ( v14 > 1u )
        break;
LABEL_17:
      v12 -= 8;
      if ( !(_DWORD)result )
        goto LABEL_18;
    }
    v15 = sub_D960E0(v13);
    v19 = (__int64 *)(v15 + 8 * v16);
    if ( (__int64 *)v15 != v19 )
    {
      v20 = (__int64 *)v15;
      while ( 2 )
      {
        v21 = *v20;
        if ( !v34 )
          goto LABEL_35;
        v22 = v31;
        v10 = HIDWORD(v32);
        v16 = (__int64)&v31[HIDWORD(v32)];
        if ( v31 != (__int64 *)v16 )
        {
          while ( v21 != *v22 )
          {
            if ( (__int64 *)v16 == ++v22 )
              goto LABEL_44;
          }
          goto LABEL_32;
        }
LABEL_44:
        if ( HIDWORD(v32) < (unsigned int)v32 )
        {
          ++HIDWORD(v32);
          *(_QWORD *)v16 = v21;
          ++v30;
LABEL_36:
          v23 = *(_WORD *)(v21 + 24);
          if ( v23 == 7 )
          {
            if ( (unsigned __int8)sub_DBE090(*(_QWORD *)a2, *(_QWORD *)(v21 + 40)) )
            {
              v23 = *(_WORD *)(v21 + 24);
              goto LABEL_39;
            }
LABEL_47:
            *(_BYTE *)(a2 + 9) = 1;
          }
          else
          {
LABEL_39:
            if ( v23 != 8 || sub_D4B130(*(_QWORD *)(v21 + 48)) )
              goto LABEL_41;
            if ( !*(_BYTE *)(a2 + 8) )
              goto LABEL_47;
            if ( *(_QWORD *)(v21 + 40) == 2 )
            {
LABEL_41:
              v24 = (unsigned int)v28;
              v25 = (unsigned int)v28 + 1LL;
              if ( v25 > HIDWORD(v28) )
              {
                sub_C8D5F0((__int64)&v27, v29, v25, 8u, v17, v18);
                v24 = (unsigned int)v28;
              }
              v16 = (__int64)v27;
              *(_QWORD *)&v27[8 * v24] = v21;
              LODWORD(v28) = v28 + 1;
            }
            else
            {
              *(_BYTE *)(a2 + 9) = 1;
            }
          }
        }
        else
        {
LABEL_35:
          sub_C8CC70((__int64)&v30, *v20, v16, v10, v17, v18);
          if ( (_BYTE)v16 )
            goto LABEL_36;
        }
LABEL_32:
        v10 = a2;
        if ( *(_BYTE *)(a2 + 9) )
          goto LABEL_34;
        if ( v19 == ++v20 )
          goto LABEL_34;
        continue;
      }
    }
    v10 = a2;
LABEL_34:
    result = (unsigned int)v28;
  }
LABEL_18:
  if ( !v34 )
  {
    result = _libc_free(v31, v12);
    v11 = v27;
  }
  if ( v11 != v29 )
    return _libc_free(v11, v12);
  return result;
}
