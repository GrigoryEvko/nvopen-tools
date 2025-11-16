// Function: sub_66A990
// Address: 0x66a990
//
void __fastcall sub_66A990(__m128i *a1, __int64 a2, __int64 a3, int a4, int a5, int a6)
{
  char v8; // dl
  __int64 v9; // rax
  int v10; // eax
  int v11; // ecx
  int v12; // r9d
  int v13; // edx
  int v14; // r8d
  __m128i *v15; // rbx
  __int8 v16; // al
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rbx
  int v21; // [rsp+4h] [rbp-3Ch]
  int v22; // [rsp+4h] [rbp-3Ch]
  int v24; // [rsp+8h] [rbp-38h]
  int v25; // [rsp+8h] [rbp-38h]
  int v27; // [rsp+Ch] [rbp-34h]
  int v28; // [rsp+Ch] [rbp-34h]

  v8 = *(_BYTE *)(a2 + 140);
  if ( v8 == 12 )
  {
    v9 = a2;
    do
    {
      v9 = *(_QWORD *)(v9 + 160);
      v8 = *(_BYTE *)(v9 + 140);
    }
    while ( v8 == 12 );
  }
  if ( !v8 )
    return;
  v10 = sub_8D3D40(a2);
  v11 = a4;
  v12 = a6;
  v13 = v10;
  if ( v10 )
  {
    if ( a1 )
      sub_684B30(1865, &a1[3].m128i_u64[1]);
    return;
  }
  if ( a1 )
  {
    v14 = 0;
    v15 = a1;
    while ( 1 )
    {
      while ( !v11 && (!a3 || (*(_BYTE *)(a3 + 130) & 8) == 0 || !a5) )
      {
        v16 = v15->m128i_i8[9];
        v15->m128i_i8[11] &= ~1u;
        if ( v16 != 1 && v16 != 4 )
        {
LABEL_14:
          if ( (v16 == 2 || (v15->m128i_i8[11] & 0x10) != 0) && v12 && v15->m128i_i8[8] > 1u )
          {
            if ( !v14 )
            {
              v21 = v12;
              v24 = v11;
              v27 = v13;
              sub_684B30((unsigned __int8)(*(_BYTE *)(a2 + 140) - 9) < 3u ? 1309 : 1571, v15[2].m128i_i64[1]);
              v12 = v21;
              v11 = v24;
              v13 = v27;
            }
            v15->m128i_i8[8] = 0;
            v14 = 1;
          }
          goto LABEL_21;
        }
LABEL_25:
        if ( a5 | v11 )
          goto LABEL_21;
        if ( !v13 )
        {
          v22 = v12;
          v25 = v11;
          v28 = v14;
          sub_6851C0(1847, &v15[3].m128i_u64[1]);
          v12 = v22;
          v11 = v25;
          v14 = v28;
        }
        v15->m128i_i8[8] = 0;
        v15 = (__m128i *)v15->m128i_i64[0];
        v13 = 1;
        if ( !v15 )
        {
LABEL_29:
          v17 = (__int64 *)a1;
          do
          {
            v17[6] = a3;
            v17 = (__int64 *)*v17;
          }
          while ( v17 );
          sub_5CEC90(a1, a2, 6);
          v18 = (__int64 *)a1;
          do
          {
            v18[6] = 0;
            v18 = (__int64 *)*v18;
          }
          while ( v18 );
          goto LABEL_33;
        }
      }
      v16 = v15->m128i_i8[9];
      v15->m128i_i8[11] |= 1u;
      if ( v16 != 1 )
      {
        if ( v16 != 4 )
          goto LABEL_14;
        goto LABEL_25;
      }
LABEL_21:
      v15 = (__m128i *)v15->m128i_i64[0];
      if ( !v15 )
        goto LABEL_29;
    }
  }
  sub_5CEC90(0, a2, 6);
LABEL_33:
  if ( a5 )
  {
    v19 = sub_86A2A0(a2);
    if ( v19 )
    {
      v20 = *(_QWORD *)(v19 + 24);
      *(_QWORD *)(v20 + 48) = sub_5CF190(a1);
    }
  }
}
