// Function: sub_11115D0
// Address: 0x11115d0
//
_QWORD *__fastcall sub_11115D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 v3; // r14
  _QWORD *v4; // r15
  __int64 v8; // rax
  bool v9; // al
  unsigned __int8 **v10; // rdx
  unsigned __int8 *v11; // rdi
  unsigned __int8 v12; // al
  __int64 v13; // r15
  __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // rbx
  _QWORD **v17; // rdx
  int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  void **v22; // rax
  void **v23; // r15
  void **v24; // r15
  unsigned __int8 **v25; // rdx
  __int64 v26; // rdx
  _BYTE *v27; // rax
  unsigned int v28; // r15d
  void **v29; // rax
  void **v30; // rdx
  char v31; // al
  _BYTE *v32; // rdx
  char v33; // [rsp+3h] [rbp-7Dh]
  int v34; // [rsp+4h] [rbp-7Ch]
  __int64 v35; // [rsp+8h] [rbp-78h]
  void **v36; // [rsp+8h] [rbp-78h]
  __int64 v37; // [rsp+18h] [rbp-68h]
  char v38[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v39; // [rsp+40h] [rbp-40h]

  v3 = *(_WORD *)(a1 + 2) & 0x3F;
  if ( (unsigned __int16)(v3 - 2) > 3u )
    return 0;
  if ( *(_BYTE *)a3 == 18 )
  {
    if ( *(void **)(a3 + 24) == sub_C33340() )
      v8 = *(_QWORD *)(a3 + 32);
    else
      v8 = a3 + 24;
    v9 = (*(_BYTE *)(v8 + 20) & 7) == 3;
  }
  else
  {
    v21 = *(_QWORD *)(a3 + 8);
    v35 = v21;
    if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 )
      return 0;
    v22 = (void **)sub_AD7630(a3, 0, v21);
    v23 = v22;
    if ( !v22 || *(_BYTE *)v22 != 18 )
    {
      if ( *(_BYTE *)(v35 + 8) == 17 )
      {
        v34 = *(_DWORD *)(v35 + 32);
        if ( v34 )
        {
          v33 = 0;
          v28 = 0;
          while ( 1 )
          {
            v29 = (void **)sub_AD69F0((unsigned __int8 *)a3, v28);
            v30 = v29;
            if ( !v29 )
              break;
            v31 = *(_BYTE *)v29;
            v36 = v30;
            if ( v31 != 13 )
            {
              if ( v31 != 18 )
                return 0;
              v32 = v30[3] == sub_C33340() ? v36[4] : v36 + 3;
              if ( (v32[20] & 7) != 3 )
                return 0;
              v33 = 1;
            }
            if ( v34 == ++v28 )
            {
              if ( v33 )
                goto LABEL_9;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    if ( v22[3] == sub_C33340() )
      v24 = (void **)v23[4];
    else
      v24 = v23 + 3;
    v9 = (*((_BYTE *)v24 + 20) & 7) == 3;
  }
  if ( !v9 )
    return 0;
LABEL_9:
  if ( !sub_B451D0(a2) || !sub_B451D0(a1) )
    return 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v10 = *(unsigned __int8 ***)(a2 - 8);
    v11 = *v10;
    v12 = **v10;
    v13 = (__int64)(*v10 + 24);
    if ( v12 == 18 )
      goto LABEL_13;
  }
  else
  {
    v25 = (unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v11 = *v25;
    v12 = **v25;
    v13 = (__int64)(*v25 + 24);
    if ( v12 == 18 )
      goto LABEL_13;
  }
  v26 = (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v11 + 1) + 8LL) - 17;
  if ( (unsigned int)v26 > 1 )
    return 0;
  if ( v12 > 0x15u )
    return 0;
  v27 = sub_AD7630((__int64)v11, 0, v26);
  if ( !v27 )
    return 0;
  v13 = (__int64)(v27 + 24);
  if ( *v27 != 18 )
    return 0;
LABEL_13:
  if ( *(void **)v13 == sub_C33340() )
  {
    v14 = *(_QWORD *)(v13 + 8);
    if ( (*(_BYTE *)(v14 + 20) & 7) != 3 )
      goto LABEL_15;
    return 0;
  }
  v14 = v13;
  if ( (*(_BYTE *)(v13 + 20) & 7) == 3 )
    return 0;
LABEL_15:
  if ( (*(_BYTE *)(v14 + 20) & 8) != 0 )
    v3 = sub_B52F50(*(_WORD *)(a1 + 2) & 0x3F);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v15 = *(_QWORD *)(a2 - 8);
  else
    v15 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v16 = *(_QWORD *)(v15 + 32);
  v39 = 257;
  v4 = sub_BD2C40(72, unk_3F10FD0);
  if ( v4 )
  {
    v17 = *(_QWORD ***)(v16 + 8);
    v18 = *((unsigned __int8 *)v17 + 8);
    if ( (unsigned int)(v18 - 17) > 1 )
    {
      v20 = sub_BCB2A0(*v17);
    }
    else
    {
      BYTE4(v37) = (_BYTE)v18 == 18;
      LODWORD(v37) = *((_DWORD *)v17 + 8);
      v19 = (__int64 *)sub_BCB2A0(*v17);
      v20 = sub_BCE1B0(v19, v37);
    }
    sub_B523C0((__int64)v4, v20, 54, v3, v16, a3, (__int64)v38, 0, 0, a1);
  }
  return v4;
}
