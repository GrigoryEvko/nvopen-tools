// Function: sub_1A24B00
// Address: 0x1a24b00
//
void __fastcall sub_1A24B00(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  _BYTE *v4; // r15
  __int64 v5; // r14
  int v6; // r8d
  int v7; // r9d
  char v8; // dl
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  _QWORD *v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // eax
  _BYTE *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r14
  char v18; // dl
  __int64 v19; // r15
  char v20; // dl
  _QWORD *v21; // r14
  _QWORD *v22; // rax
  _QWORD *v23; // rsi
  _QWORD *v24; // rcx
  __int64 v25; // rax
  _BYTE *v26; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v27; // [rsp+18h] [rbp-A8h]
  _BYTE v28[32]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v29; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v30; // [rsp+48h] [rbp-78h]
  _BYTE *v31; // [rsp+50h] [rbp-70h]
  __int64 v32; // [rsp+58h] [rbp-68h]
  int v33; // [rsp+60h] [rbp-60h]
  _BYTE v34[88]; // [rsp+68h] [rbp-58h] BYREF

  v3 = v34;
  v4 = v34;
  v5 = *(_QWORD *)(a1 + 8);
  v26 = v28;
  v27 = 0x400000000LL;
  v29 = 0;
  v30 = v34;
  v31 = v34;
  v32 = 4;
  v33 = 0;
  if ( !v5 )
    goto LABEL_30;
  while ( 1 )
  {
    v9 = sub_1648700(v5);
    if ( v4 == (_BYTE *)v3 )
    {
      v10 = &v3[HIDWORD(v32)];
      if ( v10 != v3 )
      {
        v11 = 0;
        while ( v9 != (_QWORD *)*v3 )
        {
          if ( *v3 == -2 )
            v11 = v3;
          if ( v10 == ++v3 )
          {
            if ( !v11 )
              goto LABEL_52;
            *v11 = v9;
            --v33;
            ++v29;
            goto LABEL_15;
          }
        }
        goto LABEL_4;
      }
LABEL_52:
      if ( HIDWORD(v32) < (unsigned int)v32 )
        break;
    }
    sub_16CCBA0((__int64)&v29, (__int64)v9);
    if ( v8 )
    {
LABEL_15:
      v12 = (unsigned int)v27;
      if ( (unsigned int)v27 < HIDWORD(v27) )
        goto LABEL_16;
      goto LABEL_54;
    }
LABEL_4:
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      goto LABEL_17;
LABEL_5:
    v4 = v31;
    v3 = v30;
  }
  ++HIDWORD(v32);
  *v10 = v9;
  v12 = (unsigned int)v27;
  ++v29;
  if ( (unsigned int)v27 < HIDWORD(v27) )
    goto LABEL_16;
LABEL_54:
  sub_16CD150((__int64)&v26, v28, 0, 8, v6, v7);
  v12 = (unsigned int)v27;
LABEL_16:
  *(_QWORD *)&v26[8 * v12] = v9;
  LODWORD(v27) = v27 + 1;
  v5 = *(_QWORD *)(v5 + 8);
  if ( v5 )
    goto LABEL_5;
LABEL_17:
  v13 = v27;
  v14 = v26;
  if ( (_DWORD)v27 )
  {
    while ( 2 )
    {
      while ( 2 )
      {
        v16 = v13--;
        v17 = *(_QWORD *)&v14[8 * v16 - 8];
        LODWORD(v27) = v13;
        v18 = *(_BYTE *)(v17 + 16);
        switch ( v18 )
        {
          case '6':
            v15 = *(unsigned int *)(a2 + 8);
            if ( (unsigned int)v15 >= *(_DWORD *)(a2 + 12) )
            {
              sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v6, v7);
              v15 = *(unsigned int *)(a2 + 8);
            }
            *(_QWORD *)(*(_QWORD *)a2 + 8 * v15) = v17;
            v13 = v27;
            ++*(_DWORD *)(a2 + 8);
            v14 = v26;
LABEL_22:
            if ( !v13 )
              goto LABEL_28;
            continue;
          case '7':
            goto LABEL_22;
          case '8':
            if ( !(unsigned __int8)sub_15FA1F0(v17) )
              goto LABEL_27;
            break;
          default:
            if ( (unsigned __int8)(v18 - 71) > 1u )
              goto LABEL_22;
            break;
        }
        break;
      }
      v19 = *(_QWORD *)(v17 + 8);
      if ( !v19 )
      {
LABEL_27:
        v13 = v27;
        v14 = v26;
        if ( !(_DWORD)v27 )
          goto LABEL_28;
        continue;
      }
      break;
    }
    while ( 2 )
    {
      v21 = sub_1648700(v19);
      v22 = v30;
      if ( v31 != v30 )
        goto LABEL_36;
      v23 = &v30[8 * HIDWORD(v32)];
      if ( v30 != (_BYTE *)v23 )
      {
        v24 = 0;
        while ( v21 != (_QWORD *)*v22 )
        {
          if ( *v22 == -2 )
            v24 = v22;
          if ( v23 == ++v22 )
          {
            if ( !v24 )
              goto LABEL_49;
            *v24 = v21;
            --v33;
            ++v29;
            goto LABEL_47;
          }
        }
LABEL_37:
        v19 = *(_QWORD *)(v19 + 8);
        if ( !v19 )
          goto LABEL_27;
        continue;
      }
      break;
    }
LABEL_49:
    if ( HIDWORD(v32) < (unsigned int)v32 )
    {
      ++HIDWORD(v32);
      *v23 = v21;
      v25 = (unsigned int)v27;
      ++v29;
      if ( (unsigned int)v27 >= HIDWORD(v27) )
        goto LABEL_51;
    }
    else
    {
LABEL_36:
      sub_16CCBA0((__int64)&v29, (__int64)v21);
      if ( !v20 )
        goto LABEL_37;
LABEL_47:
      v25 = (unsigned int)v27;
      if ( (unsigned int)v27 >= HIDWORD(v27) )
      {
LABEL_51:
        sub_16CD150((__int64)&v26, v28, 0, 8, v6, v7);
        v25 = (unsigned int)v27;
      }
    }
    *(_QWORD *)&v26[8 * v25] = v21;
    LODWORD(v27) = v27 + 1;
    goto LABEL_37;
  }
LABEL_28:
  if ( v14 != v28 )
    _libc_free((unsigned __int64)v14);
LABEL_30:
  if ( v31 != v30 )
    _libc_free((unsigned __int64)v31);
}
