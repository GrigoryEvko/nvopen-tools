// Function: sub_D96610
// Address: 0xd96610
//
__int64 __fastcall sub_D96610(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int16 v3; // ax
  __int64 result; // rax
  _QWORD *v5; // r8
  _QWORD *v6; // rcx
  __int64 v7; // rdi
  unsigned __int16 v8; // dx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // r12
  __int64 *v15; // r15
  __int64 v16; // rbx
  __int64 *v17; // rax
  char v18; // al
  __int16 v19; // ax
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-F0h]
  _QWORD *v23; // [rsp+8h] [rbp-E8h] BYREF
  __int64 v24; // [rsp+10h] [rbp-E0h]
  _QWORD v25[8]; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v26; // [rsp+58h] [rbp-98h] BYREF
  __int64 *v27; // [rsp+60h] [rbp-90h]
  __int64 v28; // [rsp+68h] [rbp-88h]
  int v29; // [rsp+70h] [rbp-80h]
  char v30; // [rsp+74h] [rbp-7Ch]
  __int64 v31; // [rsp+78h] [rbp-78h] BYREF

  v2 = a1 == *(_QWORD *)a2;
  v24 = 0x800000000LL;
  v27 = &v31;
  v22 = a2;
  v23 = v25;
  v28 = 0x100000008LL;
  v29 = 0;
  v30 = 1;
  v31 = a1;
  v26 = 1;
  *(_BYTE *)(a2 + 12) = v2;
  if ( !v2 && ((v3 = *(_WORD *)(a1 + 24), v3 == *(_WORD *)(a2 + 8)) || *(_WORD *)(a2 + 10) == v3 || v3 == 3) )
  {
    v25[0] = a1;
    result = 1;
    LODWORD(v24) = 1;
  }
  else
  {
    result = 0;
  }
  v5 = v25;
  while ( 1 )
  {
    v6 = &v5[(unsigned int)result];
    if ( !(_DWORD)result )
      break;
    while ( 1 )
    {
      if ( *(_BYTE *)(a2 + 12) )
        goto LABEL_12;
      v7 = *(v6 - 1);
      result = (unsigned int)(result - 1);
      LODWORD(v24) = result;
      v8 = *(_WORD *)(v7 + 24);
      if ( v8 > 0xEu )
      {
        if ( v8 != 15 )
          BUG();
        goto LABEL_11;
      }
      if ( v8 > 1u )
        break;
LABEL_11:
      --v6;
      if ( !(_DWORD)result )
        goto LABEL_12;
    }
    v9 = sub_D960E0(v7);
    v14 = (__int64 *)(v9 + 8 * v10);
    v15 = (__int64 *)v9;
    if ( (__int64 *)v9 != v14 )
    {
      while ( 1 )
      {
        v16 = *v15;
        if ( !v30 )
          goto LABEL_30;
        v17 = v27;
        v11 = HIDWORD(v28);
        v10 = (__int64)&v27[HIDWORD(v28)];
        if ( v27 != (__int64 *)v10 )
        {
          while ( v16 != *v17 )
          {
            if ( (__int64 *)v10 == ++v17 )
              goto LABEL_38;
          }
          a2 = v22;
LABEL_26:
          v18 = *(_BYTE *)(a2 + 12);
LABEL_27:
          if ( v18 )
            goto LABEL_29;
          goto LABEL_28;
        }
LABEL_38:
        if ( HIDWORD(v28) < (unsigned int)v28 )
        {
          v11 = (unsigned int)++HIDWORD(v28);
          *(_QWORD *)v10 = v16;
          a2 = v22;
          ++v26;
        }
        else
        {
LABEL_30:
          sub_C8CC70((__int64)&v26, *v15, v10, v11, v12, v13);
          a2 = v22;
          if ( !(_BYTE)v10 )
            goto LABEL_26;
        }
        v2 = v16 == *(_QWORD *)a2;
        *(_BYTE *)(a2 + 12) = v2;
        if ( v2 )
          goto LABEL_29;
        v19 = *(_WORD *)(v16 + 24);
        if ( v19 == *(_WORD *)(a2 + 8) || *(_WORD *)(a2 + 10) == v19 || v19 == 3 )
        {
          v20 = (unsigned int)v24;
          v11 = HIDWORD(v24);
          v21 = (unsigned int)v24 + 1LL;
          if ( v21 > HIDWORD(v24) )
          {
            sub_C8D5F0((__int64)&v23, v25, v21, 8u, v12, v13);
            v20 = (unsigned int)v24;
          }
          v10 = (__int64)v23;
          v23[v20] = v16;
          a2 = v22;
          LODWORD(v24) = v24 + 1;
          v18 = *(_BYTE *)(v22 + 12);
          goto LABEL_27;
        }
LABEL_28:
        if ( v14 == ++v15 )
          goto LABEL_29;
      }
    }
    a2 = v22;
LABEL_29:
    v5 = v23;
    result = (unsigned int)v24;
  }
LABEL_12:
  if ( !v30 )
  {
    result = _libc_free(v27, a2);
    v5 = v23;
  }
  if ( v5 != v25 )
    return _libc_free(v5, a2);
  return result;
}
