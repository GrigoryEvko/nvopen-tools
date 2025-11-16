// Function: sub_101E3C0
// Address: 0x101e3c0
//
unsigned __int8 *__fastcall sub_101E3C0(_BYTE *a1, unsigned __int8 *a2, char a3, __m128i *a4, int a5)
{
  __int64 v7; // r12
  unsigned __int8 v8; // al
  __int64 v9; // r15
  __int64 v11; // rsi
  unsigned int v12; // eax
  unsigned int v13; // ebx
  bool v14; // al
  unsigned __int8 v15; // al
  unsigned int v16; // edx
  unsigned __int8 v17; // al
  unsigned int v18; // edx
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rbx
  _BYTE *v22; // rax
  bool v23; // dl
  unsigned int v24; // ebx
  unsigned int v25; // ebx
  __int64 v26; // rax
  unsigned __int8 v27; // al
  __int64 *v28; // rax
  __int64 v29; // rdx
  __int64 *v30; // rax
  int v31; // [rsp+4h] [rbp-3Ch]
  bool v32; // [rsp+8h] [rbp-38h]
  bool v33; // [rsp+8h] [rbp-38h]
  int v34; // [rsp+8h] [rbp-38h]

  v7 = (__int64)a2;
  v8 = *a2;
  if ( *a1 > 0x15u )
  {
    v9 = (__int64)a1;
  }
  else
  {
    if ( v8 > 0x15u )
    {
      v9 = (__int64)a2;
      v7 = (__int64)a1;
      if ( *a1 != 13 )
        goto LABEL_7;
      return (unsigned __int8 *)v7;
    }
    v9 = sub_96E6C0(0x11u, (__int64)a1, a2, a4->m128i_i64[0]);
    if ( v9 )
      return (unsigned __int8 *)v9;
    v8 = *a2;
    v9 = (__int64)a1;
  }
  if ( v8 == 13 )
    return (unsigned __int8 *)v7;
LABEL_7:
  v11 = v7;
  if ( (unsigned __int8)sub_1003090((__int64)a4, (unsigned __int8 *)v7) )
    return (unsigned __int8 *)sub_AD6530(*(_QWORD *)(v9 + 8), v11);
  v12 = sub_FFFE90(v7);
  if ( (_BYTE)v12 )
    return (unsigned __int8 *)sub_AD6530(*(_QWORD *)(v9 + 8), v11);
  if ( *(_BYTE *)v7 == 17 )
  {
    v13 = *(_DWORD *)(v7 + 32);
    if ( v13 <= 0x40 )
      v14 = *(_QWORD *)(v7 + 24) == 1;
    else
      v14 = v13 - 1 == (unsigned int)sub_C444A0(v7 + 24);
  }
  else
  {
    v21 = *(_QWORD *)(v7 + 8);
    v32 = v12;
    if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 || *(_BYTE *)v7 > 0x15u )
      goto LABEL_16;
    v22 = sub_AD7630(v7, 0, v12);
    v23 = v32;
    if ( !v22 || *v22 != 17 )
    {
      if ( *(_BYTE *)(v21 + 8) == 17 )
      {
        v31 = *(_DWORD *)(v21 + 32);
        if ( v31 )
        {
          v25 = 0;
          while ( 1 )
          {
            v33 = v23;
            v26 = sub_AD69F0((unsigned __int8 *)v7, v25);
            if ( !v26 )
              break;
            v23 = v33;
            if ( *(_BYTE *)v26 != 13 )
            {
              if ( *(_BYTE *)v26 != 17 )
                break;
              if ( *(_DWORD *)(v26 + 32) <= 0x40u )
              {
                v23 = *(_QWORD *)(v26 + 24) == 1;
              }
              else
              {
                v34 = *(_DWORD *)(v26 + 32);
                v23 = v34 - 1 == (unsigned int)sub_C444A0(v26 + 24);
              }
              if ( !v23 )
                break;
            }
            if ( v31 == ++v25 )
            {
              if ( v23 )
                return (unsigned __int8 *)v9;
              goto LABEL_16;
            }
          }
        }
      }
      goto LABEL_16;
    }
    v24 = *((_DWORD *)v22 + 8);
    if ( v24 <= 0x40 )
      v14 = *((_QWORD *)v22 + 3) == 1;
    else
      v14 = v24 - 1 == (unsigned int)sub_C444A0((__int64)(v22 + 24));
  }
  if ( v14 )
    return (unsigned __int8 *)v9;
LABEL_16:
  if ( a4[4].m128i_i8[0] )
  {
    v15 = *(_BYTE *)v9;
    if ( *(_BYTE *)v9 > 0x1Cu )
    {
      v16 = v15 - 48;
      if ( ((unsigned __int8)(v15 - 55) <= 1u || v16 <= 1) && (*(_BYTE *)(v9 + 1) & 2) != 0 && v16 <= 1 )
      {
        v28 = (__int64 *)sub_986520(v9);
        v29 = *v28;
        if ( *v28 )
        {
          if ( v7 == v28[4] )
            return (unsigned __int8 *)v29;
        }
      }
    }
    v17 = *(_BYTE *)v7;
    if ( *(_BYTE *)v7 > 0x1Cu )
    {
      v18 = v17 - 48;
      if ( ((unsigned __int8)(v17 - 55) <= 1u || v18 <= 1) && (*(_BYTE *)(v7 + 1) & 2) != 0 && v18 <= 1 )
      {
        v30 = (__int64 *)sub_986520(v7);
        v29 = *v30;
        if ( *v30 )
        {
          if ( v9 == v30[4] )
            return (unsigned __int8 *)v29;
        }
      }
    }
  }
  v19 = *(_QWORD *)(v9 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
    v19 = **(_QWORD **)(v19 + 16);
  v11 = 1;
  if ( !sub_BCAC40(v19, 1) )
    goto LABEL_30;
  if ( a3 )
    return (unsigned __int8 *)sub_AD6530(*(_QWORD *)(v9 + 8), v11);
  if ( a5 )
  {
    v20 = sub_101D750((unsigned __int8 *)v9, (unsigned __int8 *)v7, a4, a5 - 1);
    if ( v20 )
      return (unsigned __int8 *)v20;
  }
LABEL_30:
  v20 = (__int64)sub_101B370(17, (__int64 *)v9, (__int64 *)v7, a4, a5);
  if ( v20 )
    return (unsigned __int8 *)v20;
  v20 = (__int64)sub_101C7F0(17, (__int64 *)v9, (unsigned __int8 *)v7, 0xDu, a4, a5);
  if ( v20 )
    return (unsigned __int8 *)v20;
  v27 = *(_BYTE *)v9;
  if ( *(_BYTE *)v9 != 86 && *(_BYTE *)v7 != 86 )
    goto LABEL_55;
  v20 = (__int64)sub_101C8A0(17, (__int64 *)v9, (__int64 *)v7, a4, a5);
  if ( v20 )
    return (unsigned __int8 *)v20;
  v27 = *(_BYTE *)v9;
LABEL_55:
  if ( v27 != 84 && *(_BYTE *)v7 != 84 )
    return 0;
  return sub_101CAB0(17, (__int64 *)v9, (__int64 *)v7, a4, a5);
}
