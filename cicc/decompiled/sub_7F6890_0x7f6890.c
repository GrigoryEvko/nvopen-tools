// Function: sub_7F6890
// Address: 0x7f6890
//
_QWORD *__fastcall sub_7F6890(__int64 a1, _QWORD *a2, int *a3)
{
  __m128i *v5; // rax
  __m128i *v6; // r13
  __int64 v7; // r15
  __int64 v8; // rbx
  _QWORD *v9; // rax
  __int64 *v10; // rax
  _QWORD *v11; // r15
  _BYTE *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // rax
  _QWORD *v17; // r13
  _QWORD *v18; // rax
  char v19; // al
  __int64 v20; // rdi
  __int64 *v21; // rax
  __int64 v23; // r8
  __m128i *v24; // rdi
  char v25; // al
  unsigned __int8 v26; // dl
  unsigned __int64 v27; // rsi
  unsigned __int8 v28; // r13
  __int64 v29; // rsi
  __int64 i; // rax
  _QWORD *v31; // rax
  unsigned __int8 v32; // r9
  _QWORD *v33; // rax
  __int64 *v34; // r13
  __int64 v35; // rax
  _QWORD *v36; // rax
  __int64 v37; // rsi
  __int64 v38; // rax
  __m128i *v39; // [rsp+0h] [rbp-50h]
  unsigned __int8 v40; // [rsp+0h] [rbp-50h]
  unsigned __int8 v42; // [rsp+8h] [rbp-48h]
  __m128i *v43; // [rsp+18h] [rbp-38h] BYREF

  v5 = (__m128i *)sub_724DC0();
  v6 = *(__m128i **)(a1 + 48);
  v43 = v5;
  if ( v6 )
  {
    v7 = sub_8D4050(*(_QWORD *)(a1 + 8));
    v8 = sub_691620(*(_QWORD *)(a1 + 8));
    sub_7EE560(v6, 0);
    if ( (unsigned int)sub_8D3410(v7) )
    {
      if ( (unsigned int)sub_8D28F0(v6->m128i_i64[0]) )
      {
        for ( i = v6->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v31 = sub_72BA30(*(_BYTE *)(i + 160));
        v6 = (__m128i *)sub_73E110((__int64)v6, (__int64)v31);
      }
      if ( (unsigned int)sub_8D27E0(v6->m128i_i64[0]) )
        v32 = unk_4F06A60;
      else
        v32 = byte_4F06A51[0];
      v40 = v32;
      v33 = sub_72BA30(v32);
      v34 = (__int64 *)sub_73E130(v6, (__int64)v33);
      v35 = sub_8D4490(v7);
      v36 = sub_73A8E0(v35, v40);
      v37 = *v34;
      v34[2] = (__int64)v36;
      v6 = (__m128i *)sub_73DBF0(0x29u, v37, (__int64)v34);
    }
    v9 = sub_72BA30(byte_4F06A51[0]);
    v10 = (__int64 *)sub_73E130(v6, (__int64)v9);
    v11 = v10;
    if ( a2 )
    {
      v39 = sub_7E7CA0(*v10);
      v12 = sub_731250((__int64)v39);
      v16 = (_QWORD *)sub_698020(v12, 73, (__int64)v11, v13, v14, v15);
      sub_7E69E0(v16, a3);
      v17 = sub_73E830((__int64)v39);
      v11 = sub_73E830((__int64)v39);
      v18 = sub_72BA30(byte_4F06A51[0]);
      *a2 = sub_73E130(v17, (__int64)v18);
    }
    v19 = *(_BYTE *)(v8 + 140);
    if ( v19 == 12 )
    {
      v20 = sub_8D4A00(v8);
    }
    else
    {
      if ( dword_4F077C0 && (v19 == 1 || v19 == 7) )
        goto LABEL_12;
      v20 = *(_QWORD *)(v8 + 128);
    }
    if ( v20 != 1 )
    {
      v21 = sub_73A8E0(v20, byte_4F06A51[0]);
      v11[2] = v21;
      v11 = sub_73DBF0(0x29u, *v21, (__int64)v11);
    }
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 8);
    v24 = v5;
    v25 = *(_BYTE *)(v23 + 140);
    v26 = byte_4F06A51[0];
    if ( v25 == 12 )
    {
      v42 = byte_4F06A51[0];
      v38 = sub_8D4A00(*(_QWORD *)(a1 + 8));
      v24 = v43;
      v26 = v42;
      v27 = v38;
    }
    else if ( dword_4F077C0 && (v25 == 1 || v25 == 7) )
    {
      v27 = 1;
    }
    else
    {
      v27 = *(_QWORD *)(v23 + 128);
    }
    sub_7E30B0(v24, v27, v26, 0, 0);
    v11 = sub_73A720(v43, v27);
    if ( a2 && (unsigned int)sub_8D3410(*(_QWORD *)(a1 + 8)) )
    {
      v28 = byte_4F06A51[0];
      v29 = sub_8D4490(*(_QWORD *)(a1 + 8));
      sub_7E30B0(v43, v29, v28, 0, 0);
      *a2 = sub_73A720(v43, v29);
    }
  }
LABEL_12:
  sub_724E30((__int64)&v43);
  return v11;
}
