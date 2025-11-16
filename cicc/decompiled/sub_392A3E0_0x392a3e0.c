// Function: sub_392A3E0
// Address: 0x392a3e0
//
__int64 __fastcall sub_392A3E0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _BYTE *v5; // rbx
  size_t v6; // r14
  char v7; // al
  char v8; // al
  _QWORD *v9; // rax
  __int64 v10; // r12
  __int64 v11; // rbx
  unsigned int v12; // esi
  int v13; // edx
  __m128i v14; // xmm0
  bool v15; // cc
  _QWORD *v17; // rdi
  __int64 v18; // [rsp+8h] [rbp-C8h]
  __int64 v19; // [rsp+10h] [rbp-C0h]
  __int64 v20; // [rsp+18h] [rbp-B8h]
  unsigned __int64 *v21; // [rsp+20h] [rbp-B0h]
  __int16 v22; // [rsp+2Ch] [rbp-A4h]
  char v23; // [rsp+2Eh] [rbp-A2h]
  char v24; // [rsp+2Fh] [rbp-A1h]
  __int64 v25; // [rsp+38h] [rbp-98h]
  unsigned __int64 v27[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v28[2]; // [rsp+60h] [rbp-70h] BYREF
  size_t v29; // [rsp+70h] [rbp-60h] BYREF
  __m128i v30; // [rsp+78h] [rbp-58h] BYREF
  unsigned __int64 v31; // [rsp+88h] [rbp-48h] BYREF
  unsigned int v32; // [rsp+90h] [rbp-40h]

  v5 = *(_BYTE **)(a1 + 72);
  v6 = *(_QWORD *)(a1 + 80);
  v25 = a3;
  v19 = *(_QWORD *)(a1 + 104);
  v20 = *(_QWORD *)(a1 + 144);
  v22 = *(_WORD *)(a1 + 168);
  v7 = *(_BYTE *)(a1 + 112);
  *(_BYTE *)(a1 + 112) = a4;
  v24 = v7;
  v8 = *(_BYTE *)(a1 + 171);
  *(_BYTE *)(a1 + 171) = 1;
  v23 = v8;
  v21 = (unsigned __int64 *)(a1 + 72);
  v27[0] = (unsigned __int64)v28;
  if ( &v5[v6] && !v5 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v29 = v6;
  if ( v6 > 0xF )
  {
    v27[0] = sub_22409D0((__int64)v27, &v29, 0);
    v17 = (_QWORD *)v27[0];
    v28[0] = v29;
    goto LABEL_26;
  }
  if ( v6 != 1 )
  {
    if ( !v6 )
    {
      v9 = v28;
      goto LABEL_6;
    }
    v17 = v28;
LABEL_26:
    memcpy(v17, v5, v6);
    v6 = v29;
    v9 = (_QWORD *)v27[0];
    goto LABEL_6;
  }
  LOBYTE(v28[0]) = *v5;
  v9 = v28;
LABEL_6:
  v27[1] = v6;
  *((_BYTE *)v9 + v6) = 0;
  v18 = *(_QWORD *)(a1 + 64);
  if ( !a3 )
    goto LABEL_20;
  v10 = a2 + 24;
  v11 = 0;
  while ( 1 )
  {
    (**(void (__fastcall ***)(unsigned __int64 *, __int64))a1)(&v29, a1);
    v13 = v29;
    v14 = _mm_loadu_si128(&v30);
    v15 = *(_DWORD *)(v10 + 8) <= 0x40u;
    *(_DWORD *)(v10 - 24) = v29;
    *(__m128i *)(v10 - 16) = v14;
    if ( v15 )
    {
      v12 = v32;
      if ( v32 <= 0x40 )
        break;
    }
    sub_16A51C0(v10, (__int64)&v31);
    v12 = v32;
    if ( !(_DWORD)v29 )
      goto LABEL_16;
LABEL_9:
    if ( v12 > 0x40 && v31 )
      j_j___libc_free_0_0(v31);
    ++v11;
    v10 += 40;
    if ( a3 == v11 )
      goto LABEL_20;
  }
  *(_DWORD *)(v10 + 8) = v32;
  *(_QWORD *)v10 = v31 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v12);
  if ( v13 )
    goto LABEL_9;
LABEL_16:
  if ( v12 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  v25 = v11;
LABEL_20:
  *(_QWORD *)(a1 + 64) = v18;
  sub_2240AE0(v21, v27);
  if ( (_QWORD *)v27[0] != v28 )
    j_j___libc_free_0(v27[0]);
  *(_BYTE *)(a1 + 171) = v23;
  *(_BYTE *)(a1 + 112) = v24;
  *(_WORD *)(a1 + 168) = v22;
  *(_QWORD *)(a1 + 144) = v20;
  *(_QWORD *)(a1 + 104) = v19;
  return v25;
}
