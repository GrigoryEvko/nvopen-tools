// Function: sub_DDC390
// Address: 0xddc390
//
__int64 __fastcall sub_DDC390(__int64 *a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  _BYTE **v5; // rcx
  unsigned __int64 *v6; // rsi
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rdi
  __int64 result; // rax
  unsigned int v12; // r12d
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // r13
  __int64 v16; // rbx
  __m128i *v17; // rax
  void (__fastcall *v18)(__m128i **, const __m128i **, int); // rcx
  _BYTE *v19; // r14
  _BYTE *v20; // r14
  __int64 v21; // rdx
  char v22; // [rsp+Ch] [rbp-6Ch] BYREF
  __int64 v23; // [rsp+10h] [rbp-68h] BYREF
  __int64 v24; // [rsp+18h] [rbp-60h] BYREF
  __int64 v25; // [rsp+20h] [rbp-58h] BYREF
  __m128i *v26[2]; // [rsp+28h] [rbp-50h] BYREF
  __int64 (__fastcall *v27)(__m128i **, const __m128i **, int); // [rsp+38h] [rbp-40h]
  __int64 (__fastcall *v28)(__m128i **, __int64 *); // [rsp+40h] [rbp-38h]

  v4 = *a1;
  v5 = (_BYTE **)a1[3];
  v23 = a2;
  v6 = (unsigned __int64 *)a1[1];
  v22 = a3;
  v8 = *(_QWORD *)(*(_QWORD *)v4 + 56LL);
  if ( v8 )
    v8 -= 24;
  v9 = (__int64 *)a1[2];
  v10 = a1[4];
  v24 = v8;
  result = sub_DDBDC0(v10, *v6, *v9, *v5, v23, a3, v8);
  v12 = result;
  if ( !(_BYTE)result )
  {
    result = *(unsigned __int8 *)a1[5];
    if ( (_BYTE)result )
    {
      v13 = a1[2];
      v14 = a1[3];
      v27 = 0;
      v15 = a1[4];
      v16 = a1[6];
      v17 = (__m128i *)sub_22077B0(48);
      if ( v17 )
      {
        v17->m128i_i64[0] = v13;
        v17[1].m128i_i64[0] = (__int64)&v23;
        v17[1].m128i_i64[1] = (__int64)&v22;
        v17->m128i_i64[1] = v14;
        v17[2].m128i_i64[0] = (__int64)&v24;
        v17[2].m128i_i64[1] = v15;
      }
      v26[0] = v17;
      v18 = (void (__fastcall *)(__m128i **, const __m128i **, int))sub_D91370;
      v28 = (__int64 (__fastcall *)(__m128i **, __int64 *))sub_DDD9C0;
      v27 = sub_D91370;
      v19 = *(_BYTE **)v16;
      if ( **(_BYTE **)v16 )
      {
        v20 = *(_BYTE **)(v16 + 16);
        if ( *v20 )
        {
          v12 = (unsigned __int8)*v20;
          goto LABEL_11;
        }
        LODWORD(v25) = 33;
        BYTE4(v25) = 0;
      }
      else
      {
        v25 = **(_QWORD **)(v16 + 8);
        *v19 = sub_DDD9C0(v26, &v25);
        v20 = *(_BYTE **)(v16 + 16);
        if ( *v20 )
        {
LABEL_13:
          v18 = (void (__fastcall *)(__m128i **, const __m128i **, int))v27;
          if ( **(_BYTE **)v16 && (result = **(unsigned __int8 **)(v16 + 16), (_BYTE)result) )
          {
            if ( !v27 )
              return result;
            v12 = **(unsigned __int8 **)(v16 + 16);
          }
          else if ( !v27 )
          {
            return 0;
          }
LABEL_11:
          v18(v26, (const __m128i **)v26, 3);
          return v12;
        }
        LODWORD(v25) = 33;
        BYTE4(v25) = 0;
        if ( !v27 )
          sub_4263D6(v26, &v25, v21);
      }
      *v20 = v28(v26, &v25);
      goto LABEL_13;
    }
  }
  return result;
}
