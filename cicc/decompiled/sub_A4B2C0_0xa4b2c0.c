// Function: sub_A4B2C0
// Address: 0xa4b2c0
//
__int64 __fastcall sub_A4B2C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  unsigned int v6; // r13d
  __int64 v7; // r8
  int v8; // ecx
  unsigned __int64 v9; // rdi
  char v10; // al
  _QWORD *v11; // rax
  unsigned int v13; // r9d
  _QWORD *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rbx
  int v17; // ebx
  __int64 v18; // rax
  unsigned int v19; // ebx
  char v20; // r10
  char v21; // al
  char v22; // al
  char v23; // al
  __int64 v24; // r13
  __int64 v25; // rax
  char v26; // dl
  int v27; // [rsp+4h] [rbp-8Ch]
  __int64 v28; // [rsp+8h] [rbp-88h]
  int v29; // [rsp+18h] [rbp-78h]
  unsigned int v30; // [rsp+1Ch] [rbp-74h]
  __int64 v31; // [rsp+28h] [rbp-68h] BYREF
  _QWORD *v32; // [rsp+30h] [rbp-60h] BYREF
  char v33; // [rsp+38h] [rbp-58h]
  _QWORD *v34; // [rsp+40h] [rbp-50h] BYREF
  __int64 v35; // [rsp+48h] [rbp-48h]
  _QWORD v36[8]; // [rsp+50h] [rbp-40h] BYREF

  v4 = a2;
  v6 = a3;
  sub_9C66D0((__int64)&v32, a2, a3, a4);
  v8 = v33 & 1;
  v9 = (unsigned int)(2 * v8);
  v10 = (2 * v8) | v33 & 0xFD;
  v33 = v10;
  if ( (_BYTE)v8 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    v33 = v10 & 0xFD;
    v11 = v32;
    v32 = 0;
    *(_QWORD *)a1 = v11;
LABEL_3:
    if ( v32 )
      (*(void (__fastcall **)(_QWORD *))(*v32 + 8LL))(v32);
    return a1;
  }
  v13 = v6 - 1;
  v14 = v32;
  v15 = v6 - 1;
  v16 = 1LL << ((unsigned __int8)v6 - 1);
  v29 = v16;
  if ( ((unsigned int)v32 & (unsigned int)v16) == 0 )
  {
    v22 = *(_BYTE *)(a1 + 8);
    *(_QWORD *)a1 = (unsigned int)v32;
    *(_BYTE *)(a1 + 8) = v22 & 0xFC | 2;
    return a1;
  }
  v17 = v16 - 1;
  v18 = v17 & (unsigned int)v32;
  v27 = v17;
  v19 = v6 - 1;
  v28 = v18;
  if ( v13 > 0x3F )
  {
LABEL_15:
    v24 = sub_2241E50(v9, a2, v14, v15, v7);
    v31 = 16;
    v34 = v36;
    v34 = (_QWORD *)sub_22409D0(&v34, &v31, 0);
    v36[0] = v31;
    *(__m128i *)v34 = _mm_load_si128(xmmword_3F23100);
    v35 = v31;
    *((_BYTE *)v34 + v31) = 0;
    sub_C63F00(&v31, &v34, 84, v24);
    if ( v34 != v36 )
      j_j___libc_free_0(v34, v36[0] + 1LL);
    v25 = v31;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v25 & 0xFFFFFFFFFFFFFFFELL;
    v23 = v33;
    if ( (v33 & 2) != 0 )
LABEL_22:
      sub_9CDF70(&v32);
  }
  else
  {
    while ( 1 )
    {
      a2 = v4;
      v30 = v13;
      sub_9C66D0((__int64)&v34, v4, v6, v15);
      v13 = v30;
      if ( (v33 & 2) != 0 )
        goto LABEL_22;
      if ( (v33 & 1) != 0 && v32 )
      {
        (*(void (__fastcall **)(_QWORD *))(*v32 + 8LL))(v32);
        v13 = v30;
      }
      v20 = v35 & 1;
      LOBYTE(v35) = v35 & 0xFD;
      v9 = (unsigned __int64)v34;
      v21 = v20 | v33 & 0xFE | 2;
      if ( v20 )
      {
        *(_BYTE *)(a1 + 8) |= 3u;
        v33 = v21 & 0xFD;
        *(_QWORD *)a1 = v9;
        v32 = 0;
        goto LABEL_3;
      }
      v23 = v20 & 0xFD | v33 & 0xFC;
      v15 = v19;
      v32 = v34;
      v33 = v23;
      v28 |= (unsigned __int64)((unsigned int)v34 & v27) << v19;
      if ( ((unsigned int)v34 & v29) == 0 )
        break;
      v19 += v13;
      if ( v19 > 0x3F )
        goto LABEL_15;
    }
    v26 = *(_BYTE *)(a1 + 8) & 0xFC;
    *(_QWORD *)a1 = v28;
    *(_BYTE *)(a1 + 8) = v26 | 2;
  }
  if ( (v23 & 1) != 0 )
    goto LABEL_3;
  return a1;
}
