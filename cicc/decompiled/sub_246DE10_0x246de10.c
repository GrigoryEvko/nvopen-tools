// Function: sub_246DE10
// Address: 0x246de10
//
void __fastcall sub_246DE10(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v5; // rbx
  __int64 v6; // r12
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  _BYTE *v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rsi
  unsigned __int64 v13; // rax
  _BYTE *v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  _BYTE *v18; // rax
  __int64 v19; // [rsp+8h] [rbp-148h]
  int v20; // [rsp+14h] [rbp-13Ch]
  __int64 *v21; // [rsp+30h] [rbp-120h]
  __int64 v22; // [rsp+38h] [rbp-118h]
  __m128i v23; // [rsp+40h] [rbp-110h] BYREF
  __int64 v24; // [rsp+50h] [rbp-100h]
  __int64 v25; // [rsp+58h] [rbp-F8h]
  __int64 v26; // [rsp+60h] [rbp-F0h]
  __int64 v27; // [rsp+68h] [rbp-E8h]
  __int64 v28; // [rsp+70h] [rbp-E0h]
  __int64 v29; // [rsp+78h] [rbp-D8h]
  __int16 v30; // [rsp+80h] [rbp-D0h]
  unsigned int *v31[2]; // [rsp+90h] [rbp-C0h] BYREF
  char v32; // [rsp+A0h] [rbp-B0h] BYREF
  void *v33; // [rsp+110h] [rbp-40h]

  v5 = a2;
  v19 = sub_B2BEC0(*a1);
  v20 = *(_DWORD *)(a1[1] + 4);
  v22 = a2[2];
  v21 = &a2[3 * a3];
  if ( v21 != a2 )
  {
    v6 = 0;
    while ( 1 )
    {
      sub_23D0AB0((__int64)v31, v22, 0, 0, 0);
      v10 = (_BYTE *)*v5;
      if ( *(_BYTE *)*v5 > 0x15u )
        goto LABEL_7;
      if ( (_BYTE)qword_4FE7EA8 && !sub_AD7890(*v5, v22, v7, v8, v9) )
        break;
LABEL_22:
      nullsub_61();
      v33 = &unk_49DA100;
      nullsub_63();
      if ( (char *)v31[0] != &v32 )
        _libc_free((unsigned __int64)v31[0]);
      v5 += 3;
      if ( v21 == v5 )
      {
        if ( v6 )
        {
          sub_23D0AB0((__int64)v31, v22, 0, 0, 0);
          sub_246DB90(a1, (__int64)v31, v6, 0);
          sub_F94A20(v31, (__int64)v31);
        }
        return;
      }
    }
    v30 = 257;
    v23 = (__m128i)(unsigned __int64)v19;
    v24 = 0;
    v25 = 0;
    v26 = 0;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    if ( (unsigned __int8)sub_9B6260((__int64)v10, &v23, 0) )
    {
      sub_246D400((__int64)a1, (__int64)v31, v5[1]);
      if ( !*(_BYTE *)(a1[1] + 8) )
      {
        sub_F94A20(v31, (__int64)v31);
        return;
      }
      goto LABEL_22;
    }
LABEL_7:
    if ( v20 )
    {
      sub_246DB90(a1, (__int64)v31, (__int64)v10, v5[1]);
    }
    else if ( v6 )
    {
      v23.m128i_i64[0] = (__int64)"_mscmp";
      LOWORD(v26) = 259;
      v11 = *(_QWORD *)(v6 + 8);
      if ( *(_BYTE *)(v11 + 8) != 12 )
      {
        v12 = v6;
        do
        {
          v13 = sub_24650D0((__int64)a1, v12, (__int64)v31);
          v11 = *(_QWORD *)(v13 + 8);
          v12 = v13;
        }
        while ( *(_BYTE *)(v11 + 8) != 12 );
        v6 = v13;
      }
      if ( *(_DWORD *)(v11 + 8) >> 8 != 1 )
      {
        v14 = (_BYTE *)sub_AD64C0(v11, 0, 0);
        v6 = sub_92B530(v31, 0x21u, v6, v14, (__int64)&v23);
      }
      v23.m128i_i64[0] = (__int64)"_mscmp";
      LOWORD(v26) = 259;
      v15 = *((_QWORD *)v10 + 1);
      if ( *(_BYTE *)(v15 + 8) != 12 )
      {
        v16 = (__int64)v10;
        do
        {
          v17 = sub_24650D0((__int64)a1, v16, (__int64)v31);
          v15 = *(_QWORD *)(v17 + 8);
          v16 = v17;
        }
        while ( *(_BYTE *)(v15 + 8) != 12 );
        v10 = (_BYTE *)v17;
      }
      if ( *(_DWORD *)(v15 + 8) >> 8 != 1 )
      {
        v18 = (_BYTE *)sub_AD64C0(v15, 0, 0);
        v10 = (_BYTE *)sub_92B530(v31, 0x21u, (__int64)v10, v18, (__int64)&v23);
      }
      v23.m128i_i64[0] = (__int64)"_msor";
      LOWORD(v26) = 259;
      v6 = sub_A82480(v31, (_BYTE *)v6, v10, (__int64)&v23);
    }
    else
    {
      v6 = (__int64)v10;
    }
    goto LABEL_22;
  }
}
