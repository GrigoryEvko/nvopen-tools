// Function: sub_22325B0
// Address: 0x22325b0
//
__int64 __fastcall sub_22325B0(__int64 a1, __int64 a2, _BYTE *a3, char *a4, __int64 a5, signed __int64 a6)
{
  size_t v6; // r8
  char *v7; // r14
  size_t v8; // r12
  _BYTE *v9; // rbp
  __int64 result; // rax
  __int64 v12; // r15
  size_t v13; // rdx
  __int64 v14; // rax
  _BYTE *v15; // r15
  unsigned __int8 v16; // al
  char v17; // cl
  __int64 (__fastcall *v18)(__int64, unsigned int); // rcx
  __int64 (__fastcall *v19)(__int64, unsigned int); // rcx
  __int64 (__fastcall *v20)(__int64, unsigned int); // rcx
  char v21; // r8
  char v22; // al
  char v23; // al
  size_t v25; // [rsp+8h] [rbp-40h]
  size_t v26; // [rsp+8h] [rbp-40h]
  size_t v27; // [rsp+8h] [rbp-40h]

  v6 = a5 - a6;
  v7 = a4;
  v8 = v6;
  v9 = a3;
  result = *(_DWORD *)(a1 + 24) & 0xB0;
  if ( (_DWORD)result == 32 )
  {
    if ( a6 )
    {
      result = (__int64)memcpy(a3, a4, a6);
      if ( !v8 )
        return result;
    }
    else if ( !v6 )
    {
      return result;
    }
    return (__int64)memset(&v9[a6], (char)a2, v8);
  }
  if ( (_DWORD)result != 16 )
    goto LABEL_3;
  v14 = sub_222F790((_QWORD *)(a1 + 208), a2);
  v13 = a6;
  v15 = (_BYTE *)v14;
  if ( *(_BYTE *)(v14 + 56) )
  {
    result = (unsigned __int8)*v7;
    if ( v15[102] == (_BYTE)result )
      goto LABEL_33;
    goto LABEL_18;
  }
  sub_2216D60(v14);
  v13 = a6;
  v18 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v15 + 48LL);
  result = 45;
  if ( v18 != sub_CE72A0 )
  {
    result = ((__int64 (__fastcall *)(_BYTE *, __int64, signed __int64))v18)(v15, 45, a6);
    v13 = a6;
  }
  if ( *v7 == (_BYTE)result )
    goto LABEL_33;
  if ( v15[56] )
  {
    result = (unsigned __int8)*v7;
LABEL_18:
    if ( v15[100] != (_BYTE)result )
    {
LABEL_19:
      if ( v15[105] == (_BYTE)result && a6 > 1 )
      {
LABEL_21:
        v16 = v7[1];
        if ( v15[177] == v16 )
        {
LABEL_43:
          v23 = *v7;
          v9 += 2;
          v7 += 2;
          v12 = 2;
          *(v9 - 2) = v23;
          result = (unsigned __int8)*(v7 - 1);
          *(v9 - 1) = result;
          goto LABEL_4;
        }
        goto LABEL_22;
      }
LABEL_3:
      v12 = 0;
LABEL_4:
      if ( !v8 )
      {
        v13 = a6 - v12;
        if ( a6 == v12 )
          return result;
        goto LABEL_9;
      }
      LODWORD(a2) = (char)a2;
      goto LABEL_8;
    }
LABEL_33:
    *v9 = result;
    ++v7;
    ++v9;
    v12 = 1;
    goto LABEL_4;
  }
  v25 = v13;
  sub_2216D60((__int64)v15);
  v13 = v25;
  v19 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v15 + 48LL);
  result = 43;
  if ( v19 != sub_CE72A0 )
  {
    result = ((__int64 (__fastcall *)(_BYTE *, __int64, size_t))v19)(v15, 43, v25);
    v13 = v25;
  }
  if ( *v7 == (_BYTE)result )
    goto LABEL_33;
  if ( v15[56] )
  {
    result = (unsigned __int8)*v7;
    goto LABEL_19;
  }
  v26 = v13;
  sub_2216D60((__int64)v15);
  v13 = v26;
  v20 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v15 + 48LL);
  result = 48;
  if ( v20 != sub_CE72A0 )
  {
    result = ((__int64 (__fastcall *)(_BYTE *, __int64, size_t))v20)(v15, 48, v26);
    v13 = v26;
  }
  if ( *v7 != (_BYTE)result || a6 <= 1 )
    goto LABEL_3;
  if ( v15[56] )
    goto LABEL_21;
  v27 = v13;
  v21 = sub_222EC20((__int64)v15, 0x78u);
  v16 = v7[1];
  if ( v16 == v21 )
    goto LABEL_43;
  v13 = v27;
  if ( v15[56] )
  {
LABEL_22:
    v17 = v15[145];
    goto LABEL_23;
  }
  v22 = sub_222EC20((__int64)v15, 0x58u);
  v13 = v27;
  v17 = v22;
  v16 = v7[1];
LABEL_23:
  if ( v17 == v16 )
    goto LABEL_43;
  if ( !v8 )
    return (__int64)memcpy(v9, v7, v13);
  LODWORD(a2) = (char)a2;
  v12 = 0;
LABEL_8:
  result = (__int64)memset(v9, a2, v8);
  v13 = a6 - v12;
  if ( a6 != v12 )
  {
LABEL_9:
    v9 += v8;
    return (__int64)memcpy(v9, v7, v13);
  }
  return result;
}
