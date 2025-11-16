// Function: sub_87BD00
// Address: 0x87bd00
//
char __fastcall sub_87BD00(__int64 a1, __int64 a2)
{
  __int64 i; // rbx
  __int64 v3; // r14
  _QWORD *v4; // r15
  __int64 j; // rax
  _QWORD *v6; // rax
  _QWORD *k; // rbx
  _QWORD *v8; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  char v11; // al
  __int64 v12; // r14
  __int64 v13; // rbx
  __m128i *v14; // rax
  __int64 v15; // r15
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rbx
  __m128i *v19; // rax
  __int64 v20; // rax
  __m128i *v21; // rax
  __int64 v22; // rax
  char v23; // dl
  char v24; // cl
  __int64 v25; // r15
  __m128i *v26; // rbx
  __int64 v27; // rbx
  __int64 v28; // r15
  unsigned __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rax
  char v32; // dl
  _QWORD *v34; // [rsp+8h] [rbp-48h]
  _QWORD *v35; // [rsp+10h] [rbp-40h] BYREF
  int v36[14]; // [rsp+18h] [rbp-38h] BYREF

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v3 = sub_879610("coroutine_traits");
  if ( (*(_BYTE *)(a1 + 207) & 0x10) != 0 )
  {
    v25 = *(_QWORD *)(a1 + 152);
    *(_QWORD *)(v25 + 160) = sub_72C930();
    *(_BYTE *)(a1 + 207) |= 0x20u;
    *(_BYTE *)(a2 + 120) |= 1u;
    sub_6851C0(0xAB0u, (_DWORD *)(a2 + 112));
  }
  if ( (*(_BYTE *)(a2 + 120) & 1) == 0 )
  {
    if ( !v3 )
    {
      sub_6851A0(0xA72u, (_DWORD *)(a2 + 112), (__int64)"std::coroutine_traits");
      *(_BYTE *)(a2 + 120) |= 1u;
      v12 = sub_72C930();
      goto LABEL_17;
    }
    v4 = sub_725090(0);
    v35 = v4;
    v4[4] = *(_QWORD *)(i + 160);
    for ( j = i; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    if ( *(_QWORD *)(*(_QWORD *)(j + 168) + 40LL) )
    {
      v6 = sub_725090(0);
      *v4 = v6;
      v6[4] = sub_8D71D0(i);
      v4 = (_QWORD *)*v4;
    }
    for ( k = **(_QWORD ***)(i + 168); k; v4 = (_QWORD *)*v4 )
    {
      v8 = sub_725090(0);
      *v4 = v8;
      v8[4] = k[1];
      k = (_QWORD *)*k;
    }
    v9 = sub_8AF060(v3, &v35);
    v10 = v9;
    if ( v9 )
    {
      v11 = *(_BYTE *)(v9 + 80);
      if ( v11 == 3 || dword_4F077C4 == 2 && (unsigned __int8)(v11 - 4) <= 2u )
      {
        v13 = *(_QWORD *)(v10 + 88);
        if ( v13 )
        {
          v31 = sub_879C70("promise_type", *(const char **)(v10 + 88), 0x400u);
          if ( v31 && ((v32 = *(_BYTE *)(v31 + 80), v32 == 3) || dword_4F077C4 == 2 && (unsigned __int8)(v32 - 4) <= 2u) )
          {
            v12 = *(_QWORD *)(v31 + 88);
          }
          else
          {
            sub_686A10(0x87u, (_DWORD *)(a2 + 112), (__int64)"promise_type", v10);
            v12 = sub_72C930();
          }
          goto LABEL_18;
        }
      }
    }
  }
  v12 = sub_72C930();
LABEL_17:
  v13 = sub_72C930();
LABEL_18:
  *(_QWORD *)a2 = v13;
  v14 = sub_735FB0(v12, 3, -1);
  *(_QWORD *)(a2 + 16) = v14;
  v14[4].m128i_i64[0] = *(_QWORD *)(a1 + 64);
  *(_BYTE *)(*(_QWORD *)(a2 + 16) + 89LL) |= 1u;
  v15 = sub_879610("coroutine_handle");
  if ( v15 )
  {
    *(_QWORD *)v36 = sub_725090(0);
    *(_QWORD *)(*(_QWORD *)v36 + 32LL) = v12;
    v16 = sub_8AF060(v15, v36);
    if ( v16 && ((v17 = *(_BYTE *)(v16 + 80), v17 == 3) || dword_4F077C4 == 2 && (unsigned __int8)(v17 - 4) <= 2u) )
      v18 = *(_QWORD *)(v16 + 88);
    else
      v18 = sub_72C930();
  }
  else
  {
    v18 = sub_72C930();
    sub_6851A0(0xA72u, (_DWORD *)(a2 + 112), (__int64)"std::coroutine_handle");
  }
  v19 = sub_735FB0(v18, 3, -1);
  *(_QWORD *)(a2 + 8) = v19;
  v19[4].m128i_i64[0] = *(_QWORD *)(a1 + 64);
  *(_BYTE *)(*(_QWORD *)(a2 + 8) + 89LL) |= 1u;
  v20 = sub_72C390();
  v21 = sub_735FB0(v20, 3, -1);
  *(_QWORD *)(a2 + 24) = v21;
  v21[4].m128i_i64[0] = *(_QWORD *)(a1 + 64);
  *(_BYTE *)(*(_QWORD *)(a2 + 24) + 89LL) |= 1u;
  v22 = *(_QWORD *)(a2 + 24);
  *(_BYTE *)(v22 + 177) = 3;
  v23 = *(_BYTE *)(v12 + 140);
  if ( v23 == 12 )
  {
    v22 = v12;
    do
    {
      v22 = *(_QWORD *)(v22 + 160);
      v24 = *(_BYTE *)(v22 + 140);
    }
    while ( v24 == 12 );
  }
  else
  {
    v24 = *(_BYTE *)(v12 + 140);
  }
  if ( !v24 )
    goto LABEL_27;
  while ( 1 )
  {
    LOBYTE(v22) = *(_BYTE *)(v18 + 140);
    if ( (_BYTE)v22 != 12 )
      break;
    v18 = *(_QWORD *)(v18 + 160);
  }
  if ( !(_BYTE)v22 )
  {
LABEL_27:
    *(_BYTE *)(a2 + 120) |= 1u;
    v23 = *(_BYTE *)(v12 + 140);
  }
  if ( v23 == 12 )
  {
    v22 = v12;
    do
    {
      v22 = *(_QWORD *)(v22 + 160);
      v23 = *(_BYTE *)(v22 + 140);
    }
    while ( v23 == 12 );
  }
  if ( v23 )
  {
    v27 = sub_879C70("return_value", (const char *)v12, 0x10u);
    v28 = sub_879C70("return_void", (const char *)v12, 0x10u);
    LOBYTE(v22) = v28 != 0;
    if ( v27 )
    {
      if ( v28 )
      {
        v34 = sub_67DA80(0xB9Cu, (_DWORD *)(a2 + 112), v12);
        sub_67DDB0(v34, 2973, (_QWORD *)(v27 + 48));
        sub_67DDB0(v34, 2974, (_QWORD *)(v28 + 48));
        LOBYTE(v22) = sub_685910((__int64)v34, (FILE *)0xB9E);
        goto LABEL_32;
      }
    }
    else
    {
      if ( !v28 )
        goto LABEL_32;
      v27 = v28;
    }
    if ( (*(_BYTE *)(v27 + 81) & 0x10) != 0 )
    {
      v29 = *(unsigned __int8 *)(v27 + 80);
      if ( (unsigned __int8)v29 <= 0x14u )
      {
        v30 = 1180672;
        if ( _bittest64(&v30, v29) )
        {
          LOBYTE(v22) = *(_BYTE *)(a2 + 120) & 0xFD | (2 * v22);
          *(_BYTE *)(a2 + 120) = v22;
        }
      }
    }
  }
LABEL_32:
  if ( (*(_BYTE *)(a2 + 120) & 1) == 0 )
  {
    sub_7296F0(dword_4F04C58, v36);
    v26 = sub_726410();
    sub_729730(v36[0]);
    sub_730430((__int64)v26);
    *(_QWORD *)(a2 + 48) = v26;
    LODWORD(v22) = sub_8D3D40(v12);
    if ( !(_DWORD)v22 )
      LOBYTE(v22) = sub_87AFA0((_QWORD *)a2, a1);
  }
  return v22;
}
