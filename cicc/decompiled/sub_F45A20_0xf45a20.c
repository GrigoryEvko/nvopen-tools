// Function: sub_F45A20
// Address: 0xf45a20
//
__int64 __fastcall sub_F45A20(__int64 a1, int a2, __int64 a3, unsigned __int8 *a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // r15
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  unsigned __int8 **v16; // r15
  unsigned __int8 **v17; // rax
  __int64 *v18; // rbx
  char v19; // di
  __int64 v20; // r15
  __int64 v21; // rsi
  _QWORD *v22; // rax
  __int64 *v23; // rbx
  char v24; // di
  __int64 *v25; // r13
  __int64 v26; // rsi
  _QWORD *v27; // rax
  void *v28; // [rsp+8h] [rbp-E8h]
  unsigned __int8 *v29; // [rsp+10h] [rbp-E0h]
  unsigned __int8 **v30; // [rsp+18h] [rbp-D8h]
  __int64 *v31; // [rsp+18h] [rbp-D8h]
  __int64 v32; // [rsp+20h] [rbp-D0h] BYREF
  _BYTE *v33; // [rsp+28h] [rbp-C8h]
  __int64 v34; // [rsp+30h] [rbp-C0h]
  int v35; // [rsp+38h] [rbp-B8h]
  char v36; // [rsp+3Ch] [rbp-B4h]
  _BYTE v37[176]; // [rsp+40h] [rbp-B0h] BYREF

  v28 = (void *)(a1 + 32);
  if ( a2 > 1 )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 16;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    return a1;
  }
  v8 = *(__int64 **)(a3 + 80);
  v32 = 0;
  v10 = a3;
  v33 = v37;
  v11 = *(unsigned int *)(a3 + 88);
  v34 = 16;
  v12 = (__int64)&v8[v11];
  v36 = 1;
  v35 = 0;
  if ( (__int64 *)v12 != v8 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = *v8;
        if ( (unsigned __int8 *)*v8 != a4 )
          break;
LABEL_11:
        if ( (__int64 *)v12 == ++v8 )
          goto LABEL_12;
      }
      if ( v36 )
      {
        v14 = v33;
        a3 = (__int64)&v33[8 * HIDWORD(v34)];
        if ( v33 != (_BYTE *)a3 )
        {
          while ( v13 != *v14 )
          {
            if ( (_QWORD *)a3 == ++v14 )
              goto LABEL_51;
          }
          goto LABEL_11;
        }
LABEL_51:
        if ( HIDWORD(v34) >= (unsigned int)v34 )
          goto LABEL_47;
        ++v8;
        ++HIDWORD(v34);
        *(_QWORD *)a3 = v13;
        ++v32;
        if ( (__int64 *)v12 == v8 )
          break;
      }
      else
      {
LABEL_47:
        v31 = (__int64 *)v12;
        ++v8;
        sub_C8CC70((__int64)&v32, v13, a3, v12, a5, a6);
        v12 = (__int64)v31;
        if ( v31 == v8 )
          break;
      }
    }
  }
LABEL_12:
  v15 = *(_QWORD *)(v10 + 320);
  v16 = (unsigned __int8 **)v15;
  v30 = (unsigned __int8 **)(v15 + 8LL * *(unsigned int *)(v10 + 328));
  if ( (unsigned __int8 **)v15 != v30 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        if ( (unsigned __int8)(**v16 - 18) <= 2u )
        {
          v29 = *v16;
          if ( a4 != sub_AF34D0(*v16) )
            break;
        }
LABEL_20:
        if ( v30 == ++v16 )
          goto LABEL_21;
      }
      if ( v36 )
      {
        v17 = (unsigned __int8 **)v33;
        a3 = (__int64)&v33[8 * HIDWORD(v34)];
        if ( v33 != (_BYTE *)a3 )
        {
          while ( v29 != *v17 )
          {
            if ( (unsigned __int8 **)a3 == ++v17 )
              goto LABEL_54;
          }
          goto LABEL_20;
        }
LABEL_54:
        if ( HIDWORD(v34) >= (unsigned int)v34 )
          goto LABEL_49;
        ++v16;
        ++HIDWORD(v34);
        *(_QWORD *)a3 = v29;
        ++v32;
        if ( v30 == v16 )
          break;
      }
      else
      {
LABEL_49:
        ++v16;
        sub_C8CC70((__int64)&v32, (__int64)v29, a3, v15, a5, a6);
        if ( v30 == v16 )
          break;
      }
    }
  }
LABEL_21:
  v18 = *(__int64 **)v10;
  v19 = v36;
  v20 = *(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 8);
  if ( *(_QWORD *)v10 != v20 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v21 = *v18;
        if ( v19 )
          break;
LABEL_39:
        ++v18;
        sub_C8CC70((__int64)&v32, v21, a3, v15, a5, a6);
        v19 = v36;
        if ( (__int64 *)v20 == v18 )
          goto LABEL_28;
      }
      v22 = v33;
      v15 = HIDWORD(v34);
      a3 = (__int64)&v33[8 * HIDWORD(v34)];
      if ( v33 == (_BYTE *)a3 )
      {
LABEL_44:
        if ( HIDWORD(v34) >= (unsigned int)v34 )
          goto LABEL_39;
        v15 = (unsigned int)(HIDWORD(v34) + 1);
        ++v18;
        ++HIDWORD(v34);
        *(_QWORD *)a3 = v21;
        v19 = v36;
        ++v32;
        if ( (__int64 *)v20 == v18 )
          break;
      }
      else
      {
        while ( v21 != *v22 )
        {
          if ( (_QWORD *)a3 == ++v22 )
            goto LABEL_44;
        }
        if ( (__int64 *)v20 == ++v18 )
          break;
      }
    }
  }
LABEL_28:
  v23 = *(__int64 **)(v10 + 240);
  v24 = v36;
  v25 = &v23[*(unsigned int *)(v10 + 248)];
  if ( v23 != v25 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v26 = *v23;
        if ( v24 )
          break;
LABEL_37:
        ++v23;
        sub_C8CC70((__int64)&v32, v26, a3, v15, a5, a6);
        v24 = v36;
        if ( v25 == v23 )
          goto LABEL_35;
      }
      v27 = v33;
      v15 = HIDWORD(v34);
      a3 = (__int64)&v33[8 * HIDWORD(v34)];
      if ( v33 == (_BYTE *)a3 )
      {
LABEL_41:
        if ( HIDWORD(v34) >= (unsigned int)v34 )
          goto LABEL_37;
        v15 = (unsigned int)(HIDWORD(v34) + 1);
        ++v23;
        ++HIDWORD(v34);
        *(_QWORD *)a3 = v26;
        v24 = v36;
        ++v32;
        if ( v25 == v23 )
          break;
      }
      else
      {
        while ( v26 != *v27 )
        {
          if ( (_QWORD *)a3 == ++v27 )
            goto LABEL_41;
        }
        if ( v25 == ++v23 )
          break;
      }
    }
  }
LABEL_35:
  sub_C8CF70(a1, v28, 16, (__int64)v37, (__int64)&v32);
  if ( !v36 )
    _libc_free(v33, v28);
  return a1;
}
