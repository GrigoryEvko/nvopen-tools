// Function: sub_291A1B0
// Address: 0x291a1b0
//
unsigned __int8 *__fastcall sub_291A1B0(__int64 a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // r13
  unsigned int v10; // eax
  _QWORD *v11; // rdi
  __int64 *v12; // rdx
  unsigned __int8 *v13; // r15
  __int64 v14; // rcx
  int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rax
  char v18; // dl
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // r14
  __int64 v24; // r8
  __int64 *v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // [rsp+18h] [rbp-D8h]
  __int64 v29; // [rsp+18h] [rbp-D8h]
  __int64 v30; // [rsp+30h] [rbp-C0h] BYREF
  __int64 *v31; // [rsp+38h] [rbp-B8h]
  __int64 v32; // [rsp+40h] [rbp-B0h]
  int v33; // [rsp+48h] [rbp-A8h]
  char v34; // [rsp+4Ch] [rbp-A4h]
  __int64 v35; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD *v36; // [rsp+70h] [rbp-80h] BYREF
  __int64 v37; // [rsp+78h] [rbp-78h]
  _QWORD v38[14]; // [rsp+80h] [rbp-70h] BYREF

  v31 = &v35;
  v36 = v38;
  v37 = 0x400000000LL;
  v32 = 0x100000004LL;
  v5 = *(__int64 **)(a1 + 336);
  v35 = a2;
  v33 = 0;
  v34 = 1;
  v30 = 1;
  v6 = *v5;
  v38[1] = a2;
  v38[0] = v6;
  LODWORD(v37) = 1;
  v7 = sub_B43CC0(a2);
  *a3 = 0;
  v9 = v7;
  v10 = v37;
  while ( 1 )
  {
    v11 = v36;
    v12 = &v36[2 * v10 - 2];
    v13 = (unsigned __int8 *)v12[1];
    v14 = *v12;
    LODWORD(v37) = v10 - 1;
    v15 = *v13;
    if ( (_BYTE)v15 == 61 )
    {
      v16 = *((_QWORD *)v13 + 1);
LABEL_4:
      v17 = sub_9208B0(v9, v16);
      if ( v18 )
      {
        v11 = v36;
        *(_QWORD *)(a1 + 8) = v13;
        v13 = 0;
        goto LABEL_6;
      }
      v21 = (unsigned __int64)(v17 + 7) >> 3;
      if ( v21 < *a3 )
        v21 = *a3;
      *a3 = v21;
      goto LABEL_17;
    }
    if ( (_BYTE)v15 == 62 )
    {
      v20 = *((_QWORD *)v13 - 8);
      if ( v20 == v14 )
        goto LABEL_6;
      v16 = *(_QWORD *)(v20 + 8);
      goto LABEL_4;
    }
    if ( (_BYTE)v15 == 63 )
      break;
    v22 = (unsigned int)(v15 - 78);
    if ( (unsigned __int8)(v15 - 78) > 1u && (v15 & 0xFD) != 0x54 )
      goto LABEL_6;
LABEL_22:
    v23 = *((_QWORD *)v13 + 2);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v23 + 24);
      if ( v34 )
      {
LABEL_24:
        v25 = v31;
        v14 = HIDWORD(v32);
        v22 = (unsigned __int64)&v31[HIDWORD(v32)];
        if ( v31 == (__int64 *)v22 )
          goto LABEL_37;
        while ( v24 != *v25 )
        {
          if ( (__int64 *)v22 == ++v25 )
          {
LABEL_37:
            if ( HIDWORD(v32) >= (unsigned int)v32 )
              goto LABEL_30;
            ++HIDWORD(v32);
            *(_QWORD *)v22 = v24;
            v26 = (unsigned int)v37;
            v14 = HIDWORD(v37);
            ++v30;
            v22 = (unsigned int)v37 + 1LL;
            if ( v22 <= HIDWORD(v37) )
              goto LABEL_32;
LABEL_39:
            v29 = v24;
            sub_C8D5F0((__int64)&v36, v38, v22, 0x10u, v24, v8);
            v26 = (unsigned int)v37;
            v24 = v29;
            goto LABEL_32;
          }
        }
        goto LABEL_28;
      }
      while ( 1 )
      {
LABEL_30:
        v28 = v24;
        sub_C8CC70((__int64)&v30, v24, v22, v14, v24, v8);
        v24 = v28;
        if ( (_BYTE)v22 )
        {
          v26 = (unsigned int)v37;
          v14 = HIDWORD(v37);
          v22 = (unsigned int)v37 + 1LL;
          if ( v22 > HIDWORD(v37) )
            goto LABEL_39;
LABEL_32:
          v27 = &v36[2 * v26];
          *v27 = v13;
          v27[1] = v24;
          LODWORD(v37) = v37 + 1;
          v23 = *(_QWORD *)(v23 + 8);
          if ( !v23 )
            break;
        }
        else
        {
LABEL_28:
          v23 = *(_QWORD *)(v23 + 8);
          if ( !v23 )
            break;
        }
        v24 = *(_QWORD *)(v23 + 24);
        if ( v34 )
          goto LABEL_24;
      }
    }
LABEL_17:
    v10 = v37;
    if ( !(_DWORD)v37 )
    {
      v11 = v36;
      v13 = 0;
      goto LABEL_6;
    }
  }
  if ( **(_BYTE **)(a1 + 376) )
  {
    if ( !(unsigned __int8)sub_B4DD90((__int64)v13) )
      goto LABEL_41;
    goto LABEL_22;
  }
  if ( (unsigned __int8)sub_B4DCF0((__int64)v13) )
    goto LABEL_22;
LABEL_41:
  v11 = v36;
LABEL_6:
  if ( v11 != v38 )
    _libc_free((unsigned __int64)v11);
  if ( !v34 )
    _libc_free((unsigned __int64)v31);
  return v13;
}
