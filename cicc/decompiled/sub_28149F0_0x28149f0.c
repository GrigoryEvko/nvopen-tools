// Function: sub_28149F0
// Address: 0x28149f0
//
void __fastcall sub_28149F0(__int64 a1, __int64 a2)
{
  bool v2; // zf
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  int v7; // eax
  __int64 v8; // rcx
  _BYTE *v9; // r8
  _BYTE *v10; // rsi
  __int64 v11; // rdi
  unsigned __int16 v12; // dx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // r12
  __int64 *v18; // r15
  __int64 v19; // rbx
  __int64 *v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  _BYTE *v24; // [rsp+18h] [rbp-E8h] BYREF
  __int64 v25; // [rsp+20h] [rbp-E0h]
  _BYTE v26[64]; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v27; // [rsp+68h] [rbp-98h] BYREF
  __int64 *v28; // [rsp+70h] [rbp-90h]
  __int64 v29; // [rsp+78h] [rbp-88h]
  int v30; // [rsp+80h] [rbp-80h]
  char v31; // [rsp+84h] [rbp-7Ch]
  __int64 v32; // [rsp+88h] [rbp-78h] BYREF

  v2 = *(_WORD *)(a1 + 24) == 8;
  v25 = 0x800000000LL;
  v28 = &v32;
  v24 = v26;
  v29 = 0x100000008LL;
  v30 = 0;
  v31 = 1;
  v32 = a1;
  v27 = 1;
  if ( v2 )
  {
    if ( !(unsigned __int8)sub_B19720(
                             *(_QWORD *)(*(_QWORD *)(a2 + 8) + 1224LL),
                             **(_QWORD **)(a2 + 16),
                             **(_QWORD **)(*(_QWORD *)(a1 + 48) + 32LL))
      && !(unsigned __int8)sub_B19720(
                             *(_QWORD *)(*(_QWORD *)(a2 + 8) + 1224LL),
                             **(_QWORD **)(*(_QWORD *)(a1 + 48) + 32LL),
                             **(_QWORD **)(a2 + 16)) )
    {
      *(_BYTE *)a2 = 1;
      v7 = v25;
      goto LABEL_8;
    }
    v5 = (unsigned int)v25;
    v6 = (unsigned int)v25 + 1LL;
    if ( v6 > HIDWORD(v25) )
    {
      sub_C8D5F0((__int64)&v24, v26, v6, 8u, v3, v4);
      v5 = (unsigned int)v25;
    }
  }
  else
  {
    v5 = 0;
  }
  *(_QWORD *)&v24[8 * v5] = a1;
  v7 = v25 + 1;
  LODWORD(v25) = v25 + 1;
LABEL_8:
  v8 = a2;
  while ( 1 )
  {
    v9 = v24;
    v10 = &v24[8 * v7];
    if ( !v7 )
      break;
    while ( 1 )
    {
      if ( *(_BYTE *)v8 )
        goto LABEL_14;
      v11 = *((_QWORD *)v10 - 1);
      LODWORD(v25) = --v7;
      v12 = *(_WORD *)(v11 + 24);
      if ( v12 > 0xEu )
      {
        if ( v12 != 15 )
          BUG();
        goto LABEL_13;
      }
      if ( v12 > 1u )
        break;
LABEL_13:
      v10 -= 8;
      if ( !v7 )
        goto LABEL_14;
    }
    v13 = sub_D960E0(v11);
    v17 = (__int64 *)(v13 + 8 * v14);
    v18 = (__int64 *)v13;
    if ( (__int64 *)v13 != v17 )
    {
      while ( 1 )
      {
        v19 = *v18;
        if ( v31 )
        {
          v20 = v28;
          v8 = HIDWORD(v29);
          v14 = (__int64)&v28[HIDWORD(v29)];
          if ( v28 != (__int64 *)v14 )
          {
            while ( v19 != *v20 )
            {
              if ( (__int64 *)v14 == ++v20 )
                goto LABEL_36;
            }
            goto LABEL_27;
          }
LABEL_36:
          if ( HIDWORD(v29) < (unsigned int)v29 )
            break;
        }
        sub_C8CC70((__int64)&v27, *v18, v14, v8, v15, v16);
        if ( (_BYTE)v14 )
          goto LABEL_31;
LABEL_27:
        v8 = a2;
        if ( !*(_BYTE *)a2 && v17 != ++v18 )
          continue;
        goto LABEL_29;
      }
      ++HIDWORD(v29);
      *(_QWORD *)v14 = v19;
      ++v27;
LABEL_31:
      if ( *(_WORD *)(v19 + 24) != 8
        || (unsigned __int8)sub_B19720(
                              *(_QWORD *)(*(_QWORD *)(a2 + 8) + 1224LL),
                              **(_QWORD **)(a2 + 16),
                              **(_QWORD **)(*(_QWORD *)(v19 + 48) + 32LL))
        || (unsigned __int8)sub_B19720(
                              *(_QWORD *)(*(_QWORD *)(a2 + 8) + 1224LL),
                              **(_QWORD **)(*(_QWORD *)(v19 + 48) + 32LL),
                              **(_QWORD **)(a2 + 16)) )
      {
        v21 = (unsigned int)v25;
        v22 = (unsigned int)v25 + 1LL;
        if ( v22 > HIDWORD(v25) )
        {
          sub_C8D5F0((__int64)&v24, v26, v22, 8u, v15, v16);
          v21 = (unsigned int)v25;
        }
        v14 = (__int64)v24;
        *(_QWORD *)&v24[8 * v21] = v19;
        LODWORD(v25) = v25 + 1;
      }
      else
      {
        *(_BYTE *)a2 = 1;
      }
      goto LABEL_27;
    }
    v8 = a2;
LABEL_29:
    v7 = v25;
  }
LABEL_14:
  if ( !v31 )
  {
    _libc_free((unsigned __int64)v28);
    v9 = v24;
  }
  if ( v9 != v26 )
    _libc_free((unsigned __int64)v9);
}
