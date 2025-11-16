// Function: sub_2527330
// Address: 0x2527330
//
__int64 __fastcall sub_2527330(
        __int64 a1,
        unsigned __int8 (__fastcall *a2)(__int64, _QWORD),
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6)
{
  unsigned int v7; // r13d
  unsigned __int8 *v9; // rax
  __int64 v10; // rdx
  char *v12; // r15
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r14
  char *v16; // r14
  __int64 v17; // [rsp-8h] [rbp-B8h]
  char *v20; // [rsp+18h] [rbp-98h]
  char v21; // [rsp+2Fh] [rbp-81h] BYREF
  __m128i v22; // [rsp+30h] [rbp-80h] BYREF
  char *v23; // [rsp+40h] [rbp-70h] BYREF
  __int64 v24; // [rsp+48h] [rbp-68h]
  _BYTE v25[96]; // [rsp+50h] [rbp-60h] BYREF

  v7 = 0;
  v9 = sub_250CBE0((__int64 *)(a4 + 72), (__int64)a2);
  if ( !v9 )
    return v7;
  v23 = v25;
  v24 = 0x300000000LL;
  v22.m128i_i64[0] = (unsigned __int64)v9 & 0xFFFFFFFFFFFFFFFCLL | 1;
  v21 = 0;
  v22.m128i_i64[1] = 0;
  nullsub_1518();
  v7 = sub_2526B50(a1, &v22, a4, (__int64)&v23, a5, &v21, a6);
  v10 = v17;
  if ( (_BYTE)v7 )
  {
    v12 = v23;
    v13 = 16LL * (unsigned int)v24;
    v20 = &v23[v13];
    v14 = v13 >> 4;
    v15 = v13 >> 6;
    if ( v15 )
    {
      v16 = &v23[64 * v15];
      while ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))a2)(a3, *(_QWORD *)v12, v10) )
      {
        if ( !a2(a3, *((_QWORD *)v12 + 2)) )
        {
          LOBYTE(v7) = v20 == v12 + 16;
          goto LABEL_3;
        }
        if ( !a2(a3, *((_QWORD *)v12 + 4)) )
        {
          LOBYTE(v7) = v20 == v12 + 32;
          goto LABEL_3;
        }
        if ( !a2(a3, *((_QWORD *)v12 + 6)) )
        {
          LOBYTE(v7) = v20 == v12 + 48;
          goto LABEL_3;
        }
        v12 += 64;
        if ( v16 == v12 )
        {
          v14 = (v20 - v12) >> 4;
          goto LABEL_15;
        }
      }
      goto LABEL_13;
    }
LABEL_15:
    if ( v14 != 2 )
    {
      if ( v14 != 3 )
      {
        if ( v14 != 1 )
          goto LABEL_3;
LABEL_18:
        v7 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))a2)(a3, *(_QWORD *)v12, v10);
        if ( (_BYTE)v7 )
          goto LABEL_3;
LABEL_13:
        LOBYTE(v7) = v20 == v12;
        goto LABEL_3;
      }
      if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))a2)(a3, *(_QWORD *)v12, v10) )
        goto LABEL_13;
      v12 += 16;
    }
    if ( !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))a2)(a3, *(_QWORD *)v12, v10) )
      goto LABEL_13;
    v12 += 16;
    goto LABEL_18;
  }
LABEL_3:
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
  return v7;
}
