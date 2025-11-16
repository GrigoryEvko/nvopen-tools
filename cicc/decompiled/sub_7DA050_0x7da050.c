// Function: sub_7DA050
// Address: 0x7da050
//
void __fastcall sub_7DA050(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r15
  __int64 v9; // rsi
  __int16 v10; // r14
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rdi
  __m128i *v14; // rdi
  __m128i *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  char v20; // al
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // rdi
  _QWORD *v28; // rdi
  __int64 v29; // [rsp-F0h] [rbp-F0h]
  const __m128i *v30; // [rsp-E8h] [rbp-E8h]
  __int64 *v31; // [rsp-E8h] [rbp-E8h]
  __int64 v32; // [rsp-E0h] [rbp-E0h]
  __int64 v33; // [rsp-E0h] [rbp-E0h]
  __int64 *v34; // [rsp-E0h] [rbp-E0h]
  int v35; // [rsp-D4h] [rbp-D4h]
  __int16 v36; // [rsp-CEh] [rbp-CEh]
  int v37; // [rsp-CCh] [rbp-CCh]
  int v38; // [rsp-BCh] [rbp-BCh] BYREF
  _QWORD v39[4]; // [rsp-B8h] [rbp-B8h] BYREF
  _BYTE v40[152]; // [rsp-98h] [rbp-98h] BYREF

  if ( a1 )
  {
    v6 = a1;
    v7 = a1[2].m128i_u8[8];
    v8 = unk_4D03EB0;
    unk_4D03EB0 = 0;
    v9 = dword_4F07508[0];
    v10 = dword_4F07508[1];
    v37 = dword_4D03F38[0];
    v35 = dword_4F07508[0];
    v36 = dword_4D03F38[1];
    v11 = a1->m128i_i64[0];
    v12 = (_QWORD *)a1[3].m128i_i64[0];
    *(_QWORD *)dword_4D03F38 = v11;
    *(_QWORD *)dword_4F07508 = v11;
    if ( v12 && ((unsigned __int8)v7 > 0x19u || ((1LL << v7) & 0x2003023) == 0) )
    {
      sub_7D9EC0(v12, (const __m128i *)v9, a3, v7, a5, a6);
      v7 = v6[2].m128i_u8[8];
    }
    switch ( (char)v7 )
    {
      case 0:
      case 25:
        sub_7D9920((_QWORD *)v6[3].m128i_i64[0], v6, a3, v7, a5, a6);
        sub_7E7010(v6[3].m128i_i64[0]);
        goto LABEL_10;
      case 1:
        sub_7D98E0(v6[3].m128i_i64[0], 1);
        sub_7DA050(v6[4].m128i_i64[1]);
        v13 = v6[5].m128i_i64[0];
        if ( !v13 )
          goto LABEL_10;
        goto LABEL_9;
      case 5:
      case 12:
        sub_7D98E0(v6[3].m128i_i64[0], 1);
        goto LABEL_8;
      case 6:
      case 7:
      case 8:
      case 20:
      case 22:
      case 24:
        goto LABEL_10;
      case 11:
        v21 = *(_QWORD *)(v6[5].m128i_i64[0] + 8);
        if ( v21 )
        {
          sub_7E18E0(v40, v21, 0);
          v22 = v6[4].m128i_i64[1];
          if ( !v22 )
          {
LABEL_28:
            sub_7E1AA0();
            goto LABEL_10;
          }
        }
        else
        {
          v22 = v6[4].m128i_i64[1];
          if ( !v22 )
            goto LABEL_10;
        }
        v30 = v6;
        v23 = v22;
        do
        {
          sub_7DA050(v23);
          v23 = *(_QWORD *)(v23 + 16);
        }
        while ( v23 );
        v6 = v30;
        if ( v21 )
          goto LABEL_28;
LABEL_10:
        sub_7FAF20(v6);
        LOWORD(dword_4F07508[1]) = v10;
        unk_4D03EB0 = v8;
        dword_4F07508[0] = v35;
        dword_4D03F38[0] = v37;
        LOWORD(dword_4D03F38[1]) = v36;
        break;
      case 13:
        v24 = (__int64 *)v6[5].m128i_i64[0];
        v39[0] = v6;
        v25 = (__int64)v6;
        v26 = *v24;
        if ( *v24 )
        {
          v31 = v24;
          v33 = *v24;
          sub_7DA050(*v24);
          v26 = v33;
          v24 = v31;
          if ( *(_BYTE *)(v33 + 40) || (v25 = v39[0], *(_QWORD *)(v33 + 16)) )
          {
            *v31 = 0;
            sub_7E7090(v39[0], v40, v39);
            v9 = (__int64)v40;
            sub_7E7620(v33, v40);
            v25 = v39[0];
            v24 = v31;
          }
        }
        v27 = *(_QWORD *)(v25 + 48);
        if ( v27 )
        {
          v9 = 1;
          v34 = v24;
          sub_7D98E0(v27, 1);
          v24 = v34;
        }
        v28 = (_QWORD *)v24[1];
        if ( v28 )
          sub_7D9EC0(v28, (const __m128i *)v9, v25, v7, v26, a6);
        sub_7DA050(*(_QWORD *)(v39[0] + 72LL));
        goto LABEL_10;
      case 15:
        v14 = *(__m128i **)(v6[5].m128i_i64[0] + 8);
        v32 = v6[5].m128i_i64[0];
        if ( v14 )
        {
          sub_7D8CF0(v14);
          v15 = *(__m128i **)(v32 + 16);
          if ( v15 )
            goto LABEL_17;
        }
        goto LABEL_10;
      case 16:
LABEL_8:
        v13 = v6[4].m128i_i64[1];
LABEL_9:
        sub_7DA050(v13);
        goto LABEL_10;
      case 17:
        v29 = v6[4].m128i_i64[1];
        sub_7E1720(v6, v39);
        sub_802F60(v29, 0);
        v20 = *(_BYTE *)(v29 + 48);
        switch ( v20 )
        {
          case 3:
            sub_7D9EC0(*(_QWORD **)(v29 + 56), 0, v16, v17, v18, v19);
            break;
          case 6:
            sub_7F9080(*(_QWORD *)(v29 + 8), v40);
            sub_7FEC50(v29, (unsigned int)v40, 0, 0, 1, 0, (__int64)v39, (__int64)&v38, 0);
            if ( !v38 )
              sub_7F8B60(v6);
            break;
          case 2:
            v15 = *(__m128i **)(v29 + 56);
LABEL_17:
            sub_7D8CF0(v15);
            break;
          default:
LABEL_21:
            sub_721090();
        }
        goto LABEL_10;
      case 18:
        sub_7F2990(v6);
        goto LABEL_10;
      case 21:
        sub_7DA010((__int64)v6);
        goto LABEL_10;
      case 23:
        sub_7D9920((_QWORD *)v6[3].m128i_i64[0], 0, a3, v7, a5, a6);
        sub_7E7010(v6[3].m128i_i64[0]);
        goto LABEL_10;
      default:
        goto LABEL_21;
    }
  }
}
