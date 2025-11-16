// Function: sub_1095550
// Address: 0x1095550
//
__int64 __fastcall sub_1095550(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _BYTE *v5; // rbx
  size_t v6; // r14
  char v7; // al
  char v8; // al
  _QWORD *v9; // rax
  __int64 v10; // r12
  __int64 v11; // rbx
  unsigned int v12; // ecx
  int v13; // edx
  __m128i v14; // xmm0
  bool v15; // cc
  __int64 v16; // rsi
  _QWORD *v18; // rdi
  __int64 v19; // [rsp+8h] [rbp-C8h]
  __int64 v20; // [rsp+10h] [rbp-C0h]
  __int64 v21; // [rsp+18h] [rbp-B8h]
  __int64 v22; // [rsp+20h] [rbp-B0h]
  __int16 v23; // [rsp+2Ch] [rbp-A4h]
  char v24; // [rsp+2Eh] [rbp-A2h]
  char v25; // [rsp+2Fh] [rbp-A1h]
  __int64 v26; // [rsp+38h] [rbp-98h]
  _QWORD v28[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v29[2]; // [rsp+60h] [rbp-70h] BYREF
  size_t v30; // [rsp+70h] [rbp-60h] BYREF
  __m128i v31; // [rsp+78h] [rbp-58h] BYREF
  __int64 v32; // [rsp+88h] [rbp-48h] BYREF
  unsigned int v33; // [rsp+90h] [rbp-40h]

  v5 = *(_BYTE **)(a1 + 72);
  v6 = *(_QWORD *)(a1 + 80);
  v26 = a3;
  v20 = *(_QWORD *)(a1 + 104);
  v21 = *(_QWORD *)(a1 + 152);
  v23 = *(_WORD *)(a1 + 176);
  v7 = *(_BYTE *)(a1 + 112);
  *(_BYTE *)(a1 + 112) = a4;
  v25 = v7;
  v8 = *(_BYTE *)(a1 + 178);
  *(_BYTE *)(a1 + 178) = 1;
  v24 = v8;
  v22 = a1 + 72;
  v28[0] = v29;
  if ( &v5[v6] && !v5 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v30 = v6;
  if ( v6 > 0xF )
  {
    v28[0] = sub_22409D0(v28, &v30, 0);
    v18 = (_QWORD *)v28[0];
    v29[0] = v30;
    goto LABEL_25;
  }
  if ( v6 != 1 )
  {
    if ( !v6 )
    {
      v9 = v29;
      goto LABEL_6;
    }
    v18 = v29;
LABEL_25:
    memcpy(v18, v5, v6);
    v6 = v30;
    v9 = (_QWORD *)v28[0];
    goto LABEL_6;
  }
  LOBYTE(v29[0]) = *v5;
  v9 = v29;
LABEL_6:
  v28[1] = v6;
  *((_BYTE *)v9 + v6) = 0;
  v19 = *(_QWORD *)(a1 + 64);
  if ( !a3 )
    goto LABEL_19;
  v10 = a2 + 24;
  v11 = 0;
  while ( 1 )
  {
    (**(void (__fastcall ***)(size_t *, __int64))a1)(&v30, a1);
    v13 = v30;
    v14 = _mm_loadu_si128(&v31);
    v15 = *(_DWORD *)(v10 + 8) <= 0x40u;
    *(_DWORD *)(v10 - 24) = v30;
    *(__m128i *)(v10 - 16) = v14;
    if ( v15 )
    {
      v12 = v33;
      if ( v33 <= 0x40 )
        break;
    }
    sub_C43990(v10, (__int64)&v32);
    v12 = v33;
    if ( !(_DWORD)v30 )
      goto LABEL_16;
LABEL_9:
    if ( v12 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    ++v11;
    v10 += 40;
    if ( a3 == v11 )
      goto LABEL_19;
  }
  v16 = v32;
  *(_DWORD *)(v10 + 8) = v33;
  *(_QWORD *)v10 = v16;
  if ( v13 )
    goto LABEL_9;
LABEL_16:
  v26 = v11 + 1;
  if ( v12 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
LABEL_19:
  *(_QWORD *)(a1 + 64) = v19;
  sub_2240AE0(v22, v28);
  if ( (_QWORD *)v28[0] != v29 )
    j_j___libc_free_0(v28[0], v29[0] + 1LL);
  *(_BYTE *)(a1 + 178) = v24;
  *(_BYTE *)(a1 + 112) = v25;
  *(_WORD *)(a1 + 176) = v23;
  *(_QWORD *)(a1 + 152) = v21;
  *(_QWORD *)(a1 + 104) = v20;
  return v26;
}
