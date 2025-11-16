// Function: sub_DBE8E0
// Address: 0xdbe8e0
//
__int64 __fastcall sub_DBE8E0(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int64 v4; // rsi
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
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
  __int64 *v20; // r15
  __int64 v21; // rbx
  __int64 *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  unsigned __int64 v30; // rdx
  _BYTE *v32; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v33; // [rsp+20h] [rbp-E0h]
  _BYTE v34[64]; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v35; // [rsp+68h] [rbp-98h] BYREF
  __int64 *v36; // [rsp+70h] [rbp-90h]
  __int64 v37; // [rsp+78h] [rbp-88h]
  int v38; // [rsp+80h] [rbp-80h]
  char v39; // [rsp+84h] [rbp-7Ch]
  __int64 v40; // [rsp+88h] [rbp-78h] BYREF

  v2 = *(_WORD *)(a1 + 24) == 7;
  v33 = 0x800000000LL;
  v36 = &v40;
  v32 = v34;
  v37 = 0x100000008LL;
  v38 = 0;
  v39 = 1;
  v40 = a1;
  v35 = 1;
  if ( !v2 )
  {
    v29 = 0;
LABEL_41:
    *(_QWORD *)&v32[8 * v29] = a1;
    result = (unsigned int)(v33 + 1);
    LODWORD(v33) = v33 + 1;
    goto LABEL_4;
  }
  v4 = *(_QWORD *)(a1 + 40);
  if ( (unsigned __int8)sub_DBE090(*(_QWORD *)(a2 + 8), v4)
    && (unsigned __int8)sub_D9B720(*(_QWORD *)(a1 + 40), v4, v5, v6, v7, v8) )
  {
    v29 = (unsigned int)v33;
    v30 = (unsigned int)v33 + 1LL;
    if ( HIDWORD(v33) < v30 )
    {
      sub_C8D5F0((__int64)&v32, v34, v30, 8u, v27, v28);
      v29 = (unsigned int)v33;
    }
    goto LABEL_41;
  }
  *(_BYTE *)a2 = 1;
  result = (unsigned int)v33;
LABEL_4:
  v10 = a2;
  while ( 1 )
  {
    v11 = v32;
    v12 = &v32[8 * (unsigned int)result];
    if ( !(_DWORD)result )
      break;
    while ( 1 )
    {
      if ( *(_BYTE *)v10 )
        goto LABEL_10;
      v13 = *((_QWORD *)v12 - 1);
      result = (unsigned int)(result - 1);
      LODWORD(v33) = result;
      v14 = *(_WORD *)(v13 + 24);
      if ( v14 > 0xEu )
      {
        if ( v14 != 15 )
          BUG();
        goto LABEL_9;
      }
      if ( v14 > 1u )
        break;
LABEL_9:
      v12 -= 8;
      if ( !(_DWORD)result )
        goto LABEL_10;
    }
    v15 = sub_D960E0(v13);
    v19 = (__int64 *)(v15 + 8 * v16);
    v20 = (__int64 *)v15;
    if ( (__int64 *)v15 != v19 )
    {
      while ( 1 )
      {
        v21 = *v20;
        if ( !v39 )
          goto LABEL_26;
        v22 = v36;
        v10 = HIDWORD(v37);
        v16 = (__int64)&v36[HIDWORD(v37)];
        if ( v36 != (__int64 *)v16 )
        {
          while ( v21 != *v22 )
          {
            if ( (__int64 *)v16 == ++v22 )
              goto LABEL_30;
          }
          goto LABEL_23;
        }
LABEL_30:
        if ( HIDWORD(v37) < (unsigned int)v37 )
        {
          ++HIDWORD(v37);
          *(_QWORD *)v16 = v21;
          ++v35;
          if ( *(_WORD *)(v21 + 24) != 7 )
            goto LABEL_32;
        }
        else
        {
LABEL_26:
          sub_C8CC70((__int64)&v35, *v20, v16, v10, v17, v18);
          if ( !(_BYTE)v16 )
            goto LABEL_23;
          if ( *(_WORD *)(v21 + 24) != 7 )
            goto LABEL_32;
        }
        v23 = *(_QWORD *)(v21 + 40);
        if ( (unsigned __int8)sub_DBE090(*(_QWORD *)(a2 + 8), v23)
          && (unsigned __int8)sub_D9B720(*(_QWORD *)(v21 + 40), v23, v16, v24, v17, v18) )
        {
LABEL_32:
          v25 = (unsigned int)v33;
          v26 = (unsigned int)v33 + 1LL;
          if ( v26 > HIDWORD(v33) )
          {
            sub_C8D5F0((__int64)&v32, v34, v26, 8u, v17, v18);
            v25 = (unsigned int)v33;
          }
          v16 = (__int64)v32;
          *(_QWORD *)&v32[8 * v25] = v21;
          LODWORD(v33) = v33 + 1;
          goto LABEL_23;
        }
        *(_BYTE *)a2 = 1;
LABEL_23:
        v10 = a2;
        if ( !*(_BYTE *)a2 && v19 != ++v20 )
          continue;
        goto LABEL_25;
      }
    }
    v10 = a2;
LABEL_25:
    result = (unsigned int)v33;
  }
LABEL_10:
  if ( !v39 )
  {
    result = _libc_free(v36, v12);
    v11 = v32;
  }
  if ( v11 != v34 )
    return _libc_free(v11, v12);
  return result;
}
