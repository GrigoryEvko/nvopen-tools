// Function: sub_D9B3F0
// Address: 0xd9b3f0
//
__int64 __fastcall sub_D9B3F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v6; // zf
  int v7; // eax
  _BYTE *v8; // r8
  __int64 result; // rax
  _BYTE *v10; // rsi
  _BYTE *v11; // rcx
  __int64 v12; // rdi
  unsigned __int16 v13; // dx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // r12
  __int64 *v20; // r15
  __int64 v21; // rbx
  __int64 *v22; // rax
  unsigned __int16 v23; // ax
  _QWORD *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  _QWORD *v28; // rax
  _BYTE *v30; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v31; // [rsp+20h] [rbp-E0h]
  _BYTE v32[64]; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v33; // [rsp+68h] [rbp-98h] BYREF
  __int64 *v34; // [rsp+70h] [rbp-90h]
  __int64 v35; // [rsp+78h] [rbp-88h]
  int v36; // [rsp+80h] [rbp-80h]
  char v37; // [rsp+84h] [rbp-7Ch]
  __int64 v38; // [rsp+88h] [rbp-78h] BYREF

  v6 = *(_BYTE *)a2 == 0;
  v31 = 0x800000000LL;
  v34 = &v38;
  v35 = 0x100000008LL;
  v7 = *(unsigned __int16 *)(a1 + 24);
  v30 = v32;
  v36 = 0;
  v37 = 1;
  v38 = a1;
  v33 = 1;
  if ( !v6 )
  {
LABEL_2:
    if ( (_WORD)v7 == 15 && !sub_98ED70(*(unsigned __int8 **)(a1 - 8), 0, 0, 0, 0) )
      sub_AE6EC0(a2 + 8, a1 - 32);
    sub_D9B3A0((__int64)&v30, a1, a3, a4, a5, a6);
    v8 = v30;
    result = (unsigned int)v31;
    goto LABEL_6;
  }
  if ( (_WORD)v7 != 13 )
  {
    if ( (unsigned __int16)v7 > 0xDu )
    {
      a3 = (unsigned int)(v7 - 14);
      if ( (unsigned __int16)(v7 - 14) > 1u )
LABEL_51:
        BUG();
    }
    goto LABEL_2;
  }
  result = 0;
  v8 = v32;
LABEL_6:
  v10 = &v30;
  while ( 1 )
  {
    v11 = &v8[8 * (unsigned int)result];
    if ( !(_DWORD)result )
      break;
    while ( 1 )
    {
      v12 = *((_QWORD *)v11 - 1);
      result = (unsigned int)(result - 1);
      LODWORD(v31) = result;
      v13 = *(_WORD *)(v12 + 24);
      if ( v13 <= 0xEu )
        break;
      if ( v13 != 15 )
        goto LABEL_51;
LABEL_10:
      v11 -= 8;
      if ( !(_DWORD)result )
        goto LABEL_11;
    }
    if ( v13 <= 1u )
      goto LABEL_10;
    v14 = sub_D960E0(v12);
    v19 = (__int64 *)(v14 + 8 * v15);
    v20 = (__int64 *)v14;
    if ( (__int64 *)v14 != v19 )
    {
      do
      {
        v21 = *v20;
        if ( !v37 )
          goto LABEL_26;
        v22 = v34;
        v16 = HIDWORD(v35);
        v15 = (__int64)&v34[HIDWORD(v35)];
        if ( v34 != (__int64 *)v15 )
        {
          while ( v21 != *v22 )
          {
            if ( (__int64 *)v15 == ++v22 )
              goto LABEL_33;
          }
          goto LABEL_24;
        }
LABEL_33:
        if ( HIDWORD(v35) < (unsigned int)v35 )
        {
          v16 = (unsigned int)++HIDWORD(v35);
          *(_QWORD *)v15 = v21;
          ++v33;
        }
        else
        {
LABEL_26:
          v10 = (_BYTE *)*v20;
          sub_C8CC70((__int64)&v33, *v20, v15, v16, v17, v18);
          if ( !(_BYTE)v15 )
            goto LABEL_24;
        }
        v23 = *(_WORD *)(v21 + 24);
        if ( *(_BYTE *)a2 )
          goto LABEL_28;
        if ( v23 != 13 )
        {
          if ( v23 > 0xDu && (unsigned __int16)(v23 - 14) > 1u )
            goto LABEL_51;
LABEL_28:
          if ( v23 == 15 )
          {
            v10 = 0;
            if ( !sub_98ED70(*(unsigned __int8 **)(v21 - 8), 0, 0, 0, 0) )
            {
              v10 = (_BYTE *)(v21 - 32);
              if ( !*(_BYTE *)(a2 + 36) )
                goto LABEL_45;
              v28 = *(_QWORD **)(a2 + 16);
              v25 = *(unsigned int *)(a2 + 28);
              v24 = &v28[v25];
              if ( v28 != v24 )
              {
                while ( v10 != (_BYTE *)*v28 )
                {
                  if ( v24 == ++v28 )
                    goto LABEL_43;
                }
                goto LABEL_30;
              }
LABEL_43:
              if ( (unsigned int)v25 < *(_DWORD *)(a2 + 24) )
              {
                *(_DWORD *)(a2 + 28) = v25 + 1;
                *v24 = v10;
                ++*(_QWORD *)(a2 + 8);
              }
              else
              {
LABEL_45:
                sub_C8CC70(a2 + 8, (__int64)v10, (__int64)v24, v25, v17, v18);
              }
            }
          }
LABEL_30:
          v26 = (unsigned int)v31;
          v16 = HIDWORD(v31);
          v27 = (unsigned int)v31 + 1LL;
          if ( v27 > HIDWORD(v31) )
          {
            v10 = v32;
            sub_C8D5F0((__int64)&v30, v32, v27, 8u, v17, v18);
            v26 = (unsigned int)v31;
          }
          v15 = (__int64)v30;
          *(_QWORD *)&v30[8 * v26] = v21;
          LODWORD(v31) = v31 + 1;
        }
LABEL_24:
        ++v20;
      }
      while ( v19 != v20 );
    }
    v8 = v30;
    result = (unsigned int)v31;
  }
LABEL_11:
  if ( !v37 )
  {
    result = _libc_free(v34, v10);
    v8 = v30;
  }
  if ( v8 != v32 )
    return _libc_free(v8, v10);
  return result;
}
