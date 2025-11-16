// Function: sub_388B8F0
// Address: 0x388b8f0
//
__int64 __fastcall sub_388B8F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  bool v4; // zf
  unsigned __int64 v5; // rsi
  unsigned int v6; // r13d
  __m128i v9; // xmm0
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // [rsp+8h] [rbp-B8h]
  __m128i v13; // [rsp+10h] [rbp-B0h] BYREF
  __m128i v14; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v15[2]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v16; // [rsp+40h] [rbp-80h]
  char *v17; // [rsp+50h] [rbp-70h] BYREF
  __m128i *v18; // [rsp+58h] [rbp-68h]
  __int16 v19; // [rsp+60h] [rbp-60h]
  char **v20; // [rsp+70h] [rbp-50h] BYREF
  const char *v21; // [rsp+78h] [rbp-48h]
  _WORD v22[32]; // [rsp+80h] [rbp-40h] BYREF

  v4 = *(_BYTE *)(a4 + 8) == 0;
  v13.m128i_i64[0] = a2;
  v13.m128i_i64[1] = a3;
  if ( !v4 )
  {
    v17 = "field '";
    v18 = &v13;
    v22[0] = 770;
    v5 = *(_QWORD *)(a1 + 56);
    v20 = &v17;
    v19 = 1283;
    v21 = "' cannot be specified more than once";
    return (unsigned int)sub_38814C0(a1 + 8, v5, (__int64)&v20);
  }
  v20 = (char **)v22;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  v9 = _mm_load_si128(&v13);
  v10 = *(_QWORD *)(a1 + 56);
  v21 = 0;
  LOBYTE(v22[0]) = 0;
  v12 = v10;
  v14 = v9;
  v6 = sub_388B0A0(a1, (unsigned __int64 *)&v20);
  if ( !(_BYTE)v6 )
  {
    if ( *(_BYTE *)(a4 + 9) )
    {
      if ( !v21 )
      {
        v11 = 0;
LABEL_8:
        *(_BYTE *)(a4 + 8) = 1;
        *(_QWORD *)a4 = v11;
        goto LABEL_9;
      }
LABEL_12:
      v11 = sub_161FF10(*(__int64 **)a1, v20, (size_t)v21);
      goto LABEL_8;
    }
    if ( v21 )
      goto LABEL_12;
    v19 = 1283;
    v17 = "'";
    v18 = &v14;
    v15[0] = &v17;
    v16 = 770;
    v15[1] = "' cannot be empty";
    v6 = sub_38814C0(a1 + 8, v12, (__int64)v15);
  }
LABEL_9:
  if ( v20 != (char **)v22 )
    j_j___libc_free_0((unsigned __int64)v20);
  return v6;
}
