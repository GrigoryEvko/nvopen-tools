// Function: sub_2598340
// Address: 0x2598340
//
char __fastcall sub_2598340(__int64 *a1, _QWORD *a2, char *a3)
{
  __int64 v4; // r15
  char result; // al
  int v6; // edx
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  bool (__fastcall *v12)(__int64); // rdx
  bool v13; // al
  char v14; // dl
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r14
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __m128i v24; // rax
  __m128i v25; // rax
  unsigned int v26; // [rsp+4h] [rbp-5Ch]
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+8h] [rbp-58h]
  __m128i v29; // [rsp+10h] [rbp-50h] BYREF
  __m128i v30; // [rsp+20h] [rbp-40h] BYREF

  v4 = a2[3];
  result = sub_BD2BE0(v4);
  if ( result )
    return result;
  v6 = *(unsigned __int8 *)v4;
  if ( (_BYTE)v6 != 30 && (_BYTE)v6 != 61 )
  {
    v7 = (unsigned int)(v6 - 34);
    if ( (unsigned __int8)v7 <= 0x33u
      && (v8 = 0x8000000000041LL, _bittest64(&v8, v7))
      && (v9 = *a1, v27 = a1[1], sub_254C190((unsigned __int8 *)v4, (unsigned __int64)a2))
      && *(_BYTE *)(*(_QWORD *)(*a2 + 8LL) + 8LL) == 14 )
    {
      v24.m128i_i64[0] = sub_254C9B0(v4, ((__int64)a2 - (v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))) >> 5);
      v30 = v24;
      result = sub_2588040(v9, v27, v30.m128i_i64, 1, (bool *)v29.m128i_i8, 0, 0) ^ 1;
    }
    else
    {
      result = 1;
    }
  }
  *a3 = result;
  if ( !(unsigned __int8)sub_B46420(v4) )
  {
    if ( !(unsigned __int8)sub_B46490(v4) )
      goto LABEL_16;
    v10 = a1[1];
    v28 = *a1;
    if ( *(_BYTE *)v4 != 61 )
    {
      if ( (unsigned int)*(unsigned __int8 *)v4 - 29 <= 0x20 )
      {
        if ( *(_BYTE *)v4 != 34 && *(_BYTE *)v4 != 40 )
          goto LABEL_44;
        goto LABEL_26;
      }
      if ( *(_BYTE *)v4 != 62 )
      {
        if ( *(_BYTE *)v4 != 85 )
          goto LABEL_44;
        goto LABEL_26;
      }
      goto LABEL_23;
    }
LABEL_51:
    *(_BYTE *)(v10 + 97) = *(_BYTE *)(v10 + 96) | *(_BYTE *)(v10 + 97) & 0xFE;
    goto LABEL_16;
  }
  v10 = a1[1];
  v28 = *a1;
  if ( *(_BYTE *)v4 == 61 )
    goto LABEL_51;
  if ( (unsigned int)*(unsigned __int8 *)v4 - 29 <= 0x20 )
  {
    if ( *(_BYTE *)v4 != 34 && *(_BYTE *)v4 != 40 )
      goto LABEL_14;
    goto LABEL_26;
  }
  if ( *(_BYTE *)v4 != 62 )
  {
    if ( *(_BYTE *)v4 != 85 )
      goto LABEL_14;
LABEL_26:
    if ( *(char *)(v4 + 7) >= 0 )
      goto LABEL_37;
    v15 = sub_BD2BC0(v4);
    if ( *(char *)(v4 + 7) >= 0 )
      goto LABEL_37;
    if ( !(unsigned int)((v15 + v16 - sub_BD2BC0(v4)) >> 4) )
      goto LABEL_37;
    v26 = *(_DWORD *)(v4 + 4) & 0x7FFFFFF;
    if ( *(char *)(v4 + 7) >= 0 )
      goto LABEL_37;
    v17 = sub_BD2BC0(v4);
    v19 = v17 + v18;
    if ( *(char *)(v4 + 7) >= 0 )
    {
      if ( !(unsigned int)(v19 >> 4) )
        goto LABEL_37;
    }
    else
    {
      if ( !(unsigned int)((v19 - sub_BD2BC0(v4)) >> 4) )
        goto LABEL_37;
      if ( *(char *)(v4 + 7) < 0 )
      {
        v20 = ((__int64)a2 - (v4 - 32LL * v26)) >> 5;
        if ( *(_DWORD *)(sub_BD2BC0(v4) + 8) <= (unsigned int)v20 )
        {
          if ( *(char *)(v4 + 7) >= 0 )
            BUG();
          v21 = sub_BD2BC0(v4);
          if ( *(_DWORD *)(v21 + v22 - 4) > (unsigned int)v20 )
          {
            *(_BYTE *)(v10 + 97) = *(_BYTE *)(v10 + 96);
            goto LABEL_16;
          }
        }
LABEL_37:
        if ( a2 == (_QWORD *)(v4 - 32) )
        {
          *(_BYTE *)(v10 + 97) = *(_BYTE *)(v10 + 96) | *(_BYTE *)(v10 + 97) & 0xFE;
        }
        else
        {
          v29 = 0u;
          nullsub_1518();
          if ( *(_BYTE *)(*(_QWORD *)(*a2 + 8LL) + 8LL) == 14 )
          {
            v25.m128i_i64[0] = sub_254C9B0(v4, ((__int64)a2 - (v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))) >> 5);
            v29 = v25;
          }
          else
          {
            v30.m128i_i64[1] = 0;
            v30.m128i_i64[0] = v4 & 0xFFFFFFFFFFFFFFFCLL;
            nullsub_1518();
            v29 = _mm_loadu_si128(&v30);
          }
          v23 = sub_25294B0(v28, v29.m128i_i64[0], v29.m128i_i64[1], v10, 1, 0, 1);
          if ( v23 )
          {
            *(_BYTE *)(v10 + 97) = *(_BYTE *)(v10 + 96) | *(_BYTE *)(v10 + 97) & *(_BYTE *)(v23 + 97);
            goto LABEL_16;
          }
        }
        if ( !(unsigned __int8)sub_B46420(v4) )
        {
LABEL_15:
          if ( !(unsigned __int8)sub_B46490(v4) )
            goto LABEL_16;
LABEL_44:
          *(_BYTE *)(v10 + 97) = *(_BYTE *)(v10 + 96) | *(_BYTE *)(v10 + 97) & 0xFD;
          goto LABEL_16;
        }
LABEL_14:
        *(_BYTE *)(v10 + 97) = *(_BYTE *)(v10 + 96) | *(_BYTE *)(v10 + 97) & 0xFE;
        goto LABEL_15;
      }
    }
    BUG();
  }
LABEL_23:
  v14 = *(_BYTE *)(v10 + 96);
  if ( *(_QWORD *)(v4 - 32) == *a2 )
    *(_BYTE *)(v10 + 97) = v14 | *(_BYTE *)(v10 + 97) & 0xFD;
  else
    *(_BYTE *)(v10 + 97) = v14;
LABEL_16:
  v11 = a1[1];
  v12 = *(bool (__fastcall **)(__int64))(*(_QWORD *)(v11 + 88) + 24LL);
  if ( v12 == sub_2534F80 )
    v13 = *(_BYTE *)(v11 + 97) == *(_BYTE *)(v11 + 96);
  else
    v13 = v12(v11 + 88);
  return !v13;
}
