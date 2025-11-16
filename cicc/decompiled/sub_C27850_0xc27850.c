// Function: sub_C27850
// Address: 0xc27850
//
__int64 __fastcall sub_C27850(__int64 a1, unsigned __int8 a2, __int64 a3)
{
  _QWORD *v3; // r15
  __int64 result; // rax
  _QWORD *v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // r14
  size_t v9; // r15
  void *v10; // r13
  _QWORD *v11; // rbx
  size_t v12; // rdx
  int v13; // eax
  size_t v14; // r12
  const void *v15; // rdi
  _QWORD *v16; // rax
  _QWORD *v17; // r14
  const void *v18; // rbx
  size_t v19; // r13
  size_t v20; // rcx
  const void *v21; // rsi
  size_t v22; // rdx
  int v23; // eax
  _QWORD *v24; // [rsp+8h] [rbp-128h]
  _QWORD *v25; // [rsp+10h] [rbp-120h]
  __int64 v26; // [rsp+18h] [rbp-118h]
  size_t v27; // [rsp+18h] [rbp-118h]
  _QWORD *v28; // [rsp+20h] [rbp-110h]
  int v29; // [rsp+38h] [rbp-F8h]
  const __m128i *v31; // [rsp+40h] [rbp-F0h] BYREF
  unsigned int v32[2]; // [rsp+48h] [rbp-E8h] BYREF
  _QWORD v33[2]; // [rsp+50h] [rbp-E0h] BYREF
  unsigned int v34; // [rsp+60h] [rbp-D0h] BYREF
  char v35; // [rsp+70h] [rbp-C0h]
  __int64 v36; // [rsp+80h] [rbp-B0h] BYREF
  char v37; // [rsp+90h] [rbp-A0h]
  __int64 v38; // [rsp+A0h] [rbp-90h] BYREF
  char v39; // [rsp+B0h] [rbp-80h]
  void *s2; // [rsp+C0h] [rbp-70h] BYREF
  size_t n; // [rsp+C8h] [rbp-68h]
  char v42; // [rsp+D0h] [rbp-60h]
  char v43; // [rsp+F0h] [rbp-40h]

  if ( *(_QWORD *)(a1 + 208) >= *(_QWORD *)(a1 + 216) )
  {
LABEL_5:
    sub_C1AFD0();
    return 0;
  }
  v3 = (_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 177) )
  {
    sub_C21E40((__int64)&s2, (_QWORD *)a1);
    if ( (v42 & 1) != 0 )
    {
      result = (unsigned int)s2;
      if ( (_DWORD)s2 )
        return result;
    }
    if ( a3 )
      *(_QWORD *)(a3 + 8) = s2;
  }
  if ( a2 )
  {
    sub_C22200((__int64)&s2, (_QWORD *)a1);
    if ( (v42 & 1) != 0 )
    {
      result = (unsigned int)s2;
      if ( (_DWORD)s2 )
        return result;
    }
    if ( a3 )
      *(_DWORD *)(a3 + 52) = (_DWORD)s2;
  }
  if ( *(_BYTE *)(a1 + 178) )
    goto LABEL_5;
  sub_C22200((__int64)&v34, (_QWORD *)a1);
  if ( (v35 & 1) == 0 || (result = v34) == 0 )
  {
    if ( !v34 )
      goto LABEL_5;
    v28 = (_QWORD *)a3;
    v29 = 0;
    while ( 1 )
    {
      sub_C21E40((__int64)&v36, v3);
      if ( (v37 & 1) != 0 )
      {
        result = (unsigned int)v36;
        if ( (_DWORD)v36 )
          return result;
      }
      sub_C21E40((__int64)&v38, v3);
      if ( (v39 & 1) != 0 )
      {
        result = (unsigned int)v38;
        if ( (_DWORD)v38 )
          return result;
      }
      sub_C22680((__int64)&s2, (__int64)v3);
      if ( (v43 & 1) != 0 )
      {
        result = (unsigned int)s2;
        if ( (_DWORD)s2 )
          return result;
      }
      v6 = 0;
      if ( !v28 )
        goto LABEL_52;
      v32[0] = v36;
      v32[1] = v38;
      v7 = sub_C273A0(v28, v32);
      v8 = *(_QWORD **)(v7 + 16);
      v24 = (_QWORD *)v7;
      v33[0] = s2;
      v33[1] = n;
      v26 = v7 + 8;
      if ( !v8 )
      {
        v17 = (_QWORD *)(v7 + 8);
        goto LABEL_56;
      }
      v25 = v3;
      v9 = n;
      v10 = s2;
      v11 = (_QWORD *)(v7 + 8);
      do
      {
        while ( 1 )
        {
          v14 = v8[5];
          v15 = (const void *)v8[4];
          if ( v9 < v14 )
            break;
          if ( v10 == v15 )
            goto LABEL_32;
          v12 = v8[5];
LABEL_29:
          if ( !v15 )
            goto LABEL_41;
          if ( !v10 )
            goto LABEL_34;
          v13 = memcmp(v15, v10, v12);
          if ( v13 )
          {
            if ( v13 >= 0 )
              goto LABEL_34;
            goto LABEL_41;
          }
LABEL_32:
          if ( v9 != v14 )
            goto LABEL_33;
LABEL_34:
          v11 = v8;
          v8 = (_QWORD *)v8[2];
          if ( !v8 )
            goto LABEL_42;
        }
        if ( v10 != v15 )
        {
          v12 = v9;
          goto LABEL_29;
        }
LABEL_33:
        if ( v9 <= v14 )
          goto LABEL_34;
LABEL_41:
        v8 = (_QWORD *)v8[3];
      }
      while ( v8 );
LABEL_42:
      v16 = v11;
      v17 = v11;
      v18 = v10;
      v19 = v9;
      v3 = v25;
      if ( (_QWORD *)v26 == v16 )
        goto LABEL_56;
      v20 = v16[5];
      v21 = (const void *)v16[4];
      if ( v19 > v20 )
      {
        if ( v18 == v21 )
          goto LABEL_50;
        v22 = v16[5];
LABEL_46:
        v27 = v16[5];
        if ( !v18 )
          goto LABEL_56;
        if ( v21 )
        {
          v23 = memcmp(v18, v21, v22);
          v20 = v27;
          if ( !v23 )
            goto LABEL_49;
          if ( v23 < 0 )
            goto LABEL_56;
        }
      }
      else
      {
        if ( v18 != v21 )
        {
          v22 = v19;
          goto LABEL_46;
        }
LABEL_49:
        if ( v19 != v20 )
        {
LABEL_50:
          if ( v19 >= v20 )
            goto LABEL_51;
LABEL_56:
          v31 = (const __m128i *)v33;
          v17 = (_QWORD *)sub_C275A0(v24, v17, &v31);
        }
      }
LABEL_51:
      v6 = v17 + 6;
LABEL_52:
      result = sub_C27850(v3, a2, v6);
      if ( (_DWORD)result )
        return result;
      if ( v34 <= ++v29 )
        goto LABEL_5;
    }
  }
  return result;
}
