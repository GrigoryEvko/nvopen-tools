// Function: sub_302B950
// Address: 0x302b950
//
void __fastcall sub_302B950(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  unsigned int v7; // eax
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  _QWORD *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rdi
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned int v16; // [rsp+Ch] [rbp-1C4h]
  _QWORD v17[4]; // [rsp+10h] [rbp-1C0h] BYREF
  __int16 v18; // [rsp+30h] [rbp-1A0h]
  _QWORD v19[3]; // [rsp+40h] [rbp-190h] BYREF
  __int64 v20; // [rsp+58h] [rbp-178h]
  __m128i *v21; // [rsp+60h] [rbp-170h]
  __int64 v22; // [rsp+68h] [rbp-168h]
  unsigned __int64 *v23; // [rsp+70h] [rbp-160h]
  _QWORD v24[2]; // [rsp+80h] [rbp-150h] BYREF
  _BYTE *v25; // [rsp+90h] [rbp-140h]
  __int64 v26; // [rsp+98h] [rbp-138h]
  _BYTE v27[96]; // [rsp+A0h] [rbp-130h] BYREF
  unsigned __int64 v28[3]; // [rsp+100h] [rbp-D0h] BYREF
  _BYTE v29[184]; // [rsp+118h] [rbp-B8h] BYREF

  sub_31DB000();
  nullsub_1705();
  v25 = v27;
  v24[0] = 0;
  v24[1] = 0;
  v26 = 0x600000000LL;
  sub_302B7B0(a1, a2, v24, v3, v4, v5);
  v6 = sub_30223B0(a1, a2);
  if ( v6 )
  {
    v7 = *(_DWORD *)(v6 + 80);
    v28[1] = 0;
    v16 = v7;
    v22 = 0x100000000LL;
    v19[0] = &unk_49DD288;
    v23 = v28;
    v28[0] = (unsigned __int64)v29;
    v28[2] = 128;
    v19[1] = 2;
    v19[2] = 0;
    v20 = 0;
    v21 = 0;
    sub_CB5980((__int64)v19, 0, 0, 0);
    v8 = v21;
    if ( (unsigned __int64)(v20 - (_QWORD)v21) <= 0x19 )
    {
      v10 = (_QWORD *)sub_CB6200((__int64)v19, "\t.pragma \"used_bytes_mask ", 0x1Au);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_4327160);
      v21[1].m128i_i16[4] = 8299;
      v10 = v19;
      v8[1].m128i_i64[0] = 0x73616D5F73657479LL;
      *v8 = si128;
      v21 = (__m128i *)((char *)v21 + 26);
    }
    v11 = sub_CB59D0((__int64)v10, v16);
    v12 = *(_QWORD *)(v11 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v11 + 24) - v12) <= 2 )
    {
      sub_CB6200(v11, (unsigned __int8 *)"\";\n", 3u);
    }
    else
    {
      *(_BYTE *)(v12 + 2) = 10;
      *(_WORD *)v12 = 15138;
      *(_QWORD *)(v11 + 32) += 3LL;
    }
    v13 = *(__int64 **)(a1 + 224);
    v14 = v23[1];
    v15 = *v23;
    v18 = 261;
    v17[0] = v15;
    v17[1] = v14;
    sub_E99A90(v13, (__int64)v17);
    v19[0] = &unk_49DD388;
    sub_CB5840((__int64)v19);
    if ( (_BYTE *)v28[0] != v29 )
      _libc_free(v28[0]);
  }
  sub_31DB460(a1, *(_QWORD *)(a1 + 224), v24);
  if ( v25 != v27 )
    _libc_free((unsigned __int64)v25);
}
