// Function: sub_2538D20
// Address: 0x2538d20
//
_QWORD *__fastcall sub_2538D20(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __m128i v4; // rcx
  __int64 v5; // r12
  __m128i v8; // rax
  char v9; // al
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // r14
  _QWORD *v15; // rbx
  __int64 v16; // r10
  __int64 v17; // r12
  __int64 v18; // r13
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 v21; // rdx
  int v22; // eax
  char v23; // cl
  int v24; // eax
  __m128i v25; // xmm1
  __int64 v26; // [rsp+8h] [rbp-108h] BYREF
  __int64 v27; // [rsp+10h] [rbp-100h] BYREF
  __int64 v28; // [rsp+18h] [rbp-F8h]
  __m128i v29; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v30; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+40h] [rbp-D0h]
  __m128i v32; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v33; // [rsp+70h] [rbp-A0h]
  __m128i v34; // [rsp+80h] [rbp-90h] BYREF
  __m128i v35; // [rsp+90h] [rbp-80h]
  __int64 v36; // [rsp+A0h] [rbp-70h]
  char v37[32]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v38; // [rsp+D0h] [rbp-40h]

  v4.m128i_i64[0] = a4;
  v5 = a1;
  v26 = a2;
  if ( a2 )
  {
    v33 = 268;
    v32.m128i_i64[0] = (__int64)&v26;
    v8.m128i_i64[0] = (__int64)sub_BD5D20(a1);
    v29 = v8;
    v30.m128i_i64[0] = (__int64)".b";
    v9 = v33;
    LOWORD(v31) = 773;
    if ( (_BYTE)v33 )
    {
      if ( (_BYTE)v33 == 1 )
      {
        v25 = _mm_loadu_si128(&v30);
        v34 = _mm_loadu_si128(&v29);
        v36 = v31;
        v35 = v25;
      }
      else
      {
        if ( HIBYTE(v33) == 1 )
        {
          v4 = v32;
        }
        else
        {
          v4.m128i_i64[0] = (__int64)&v32;
          v9 = 2;
        }
        v35 = v4;
        v34.m128i_i64[0] = (__int64)&v29;
        LOBYTE(v36) = 2;
        BYTE1(v36) = v9;
      }
    }
    else
    {
      LOWORD(v36) = 256;
    }
    v10 = v26;
    v11 = sub_BCB2E0((_QWORD *)a3[9]);
    v12 = sub_ACD640(v11, v10, 0);
    v13 = (_QWORD *)a3[9];
    v27 = v12;
    v14 = sub_BCB2B0(v13);
    v15 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *, __int64, _QWORD))(*(_QWORD *)a3[10] + 64LL))(
                      a3[10],
                      v14,
                      v5,
                      &v27,
                      1,
                      0);
    if ( v15 )
      return v15;
    v38 = 257;
    v15 = sub_BD2C40(88, 2u);
    if ( !v15 )
    {
LABEL_14:
      sub_B4DDE0((__int64)v15, 0);
      (*(void (__fastcall **)(__int64, _QWORD *, __m128i *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
        a3[11],
        v15,
        &v34,
        a3[7],
        a3[8]);
      v17 = *a3;
      v18 = *a3 + 16LL * *((unsigned int *)a3 + 2);
      while ( v18 != v17 )
      {
        v19 = *(_QWORD *)(v17 + 8);
        v20 = *(_DWORD *)v17;
        v17 += 16;
        sub_B99FD0((__int64)v15, v20, v19);
      }
      return v15;
    }
    v16 = *(_QWORD *)(v5 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 <= 1 )
    {
LABEL_13:
      sub_B44260((__int64)v15, v16, 34, 2u, 0, 0);
      v15[9] = v14;
      v15[10] = sub_B4DC50(v14, (__int64)&v27, 1);
      sub_B4D9A0((__int64)v15, v5, &v27, 1, (__int64)v37);
      goto LABEL_14;
    }
    v21 = *(_QWORD *)(v27 + 8);
    v22 = *(unsigned __int8 *)(v21 + 8);
    if ( v22 == 17 )
    {
      v23 = 0;
    }
    else
    {
      v23 = 1;
      if ( v22 != 18 )
        goto LABEL_13;
    }
    v24 = *(_DWORD *)(v21 + 32);
    BYTE4(v28) = v23;
    LODWORD(v28) = v24;
    v16 = sub_BCE1B0((__int64 *)v16, v28);
    goto LABEL_13;
  }
  return (_QWORD *)v5;
}
