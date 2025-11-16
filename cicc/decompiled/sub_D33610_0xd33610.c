// Function: sub_D33610
// Address: 0xd33610
//
__int64 __fastcall sub_D33610(
        __int64 a1,
        __int64 a2,
        void (__fastcall *a3)(__int64, __int64),
        __int64 a4,
        __int64 a5,
        _QWORD *a6)
{
  __int64 v8; // rbx
  _QWORD *v9; // rdi
  _QWORD *v10; // rsi
  __int64 result; // rax
  __int64 v12; // rdx
  __int64 v13; // r15
  char *v14; // rax
  char v15; // dl
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // [rsp+0h] [rbp-F0h]
  _QWORD *v21; // [rsp+8h] [rbp-E8h]
  _QWORD *v22; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v23; // [rsp+28h] [rbp-C8h]
  _QWORD v24[6]; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+60h] [rbp-90h] BYREF
  char *v26; // [rsp+68h] [rbp-88h]
  __int64 v27; // [rsp+70h] [rbp-80h]
  int v28; // [rsp+78h] [rbp-78h]
  char v29; // [rsp+7Ch] [rbp-74h]
  char v30; // [rsp+80h] [rbp-70h] BYREF

  v8 = a4;
  v22 = v24;
  v24[0] = a1;
  v9 = v24;
  v10 = &v22;
  v25 = 0;
  v27 = 8;
  v28 = 0;
  v29 = 1;
  v26 = &v30;
  v23 = 0x600000001LL;
  result = 1;
  while ( (_DWORD)result )
  {
    while ( 1 )
    {
      v12 = (unsigned int)result;
      v13 = v9[(unsigned int)result - 1];
      LODWORD(v23) = result - 1;
      if ( v29 )
      {
        v14 = v26;
        a4 = HIDWORD(v27);
        v12 = (__int64)&v26[8 * HIDWORD(v27)];
        if ( v26 != (char *)v12 )
        {
          while ( v13 != *(_QWORD *)v14 )
          {
            v14 += 8;
            if ( (char *)v12 == v14 )
              goto LABEL_17;
          }
          goto LABEL_8;
        }
LABEL_17:
        if ( HIDWORD(v27) < (unsigned int)v27 )
          break;
      }
      v10 = (_QWORD *)v13;
      sub_C8CC70((__int64)&v25, v13, v12, a4, a5, (__int64)a6);
      if ( v15 )
      {
        if ( *(_BYTE *)v13 == 84 )
          goto LABEL_19;
        goto LABEL_15;
      }
LABEL_8:
      result = (unsigned int)v23;
      v9 = v22;
      if ( !(_DWORD)v23 )
        goto LABEL_9;
    }
    ++HIDWORD(v27);
    *(_QWORD *)v12 = v13;
    ++v25;
    if ( *(_BYTE *)v13 != 84 )
      goto LABEL_15;
LABEL_19:
    v10 = *(_QWORD **)(v13 + 40);
    if ( !*(_BYTE *)(a2 + 84) )
    {
      if ( sub_C8CA60(a2 + 56, (__int64)v10) )
      {
        v17 = *(_QWORD **)(v13 + 40);
        goto LABEL_24;
      }
LABEL_15:
      v10 = (_QWORD *)v13;
      a3(v8, v13);
      result = (unsigned int)v23;
      goto LABEL_16;
    }
    v16 = *(_QWORD **)(a2 + 64);
    a4 = (__int64)&v16[*(unsigned int *)(a2 + 76)];
    if ( v16 == (_QWORD *)a4 )
      goto LABEL_15;
    while ( 1 )
    {
      v17 = (_QWORD *)*v16;
      if ( v10 == (_QWORD *)*v16 )
        break;
      if ( (_QWORD *)a4 == ++v16 )
        goto LABEL_15;
    }
LABEL_24:
    if ( **(_QWORD ***)(a2 + 32) == v17 )
      goto LABEL_15;
    v18 = 4LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
    {
      a6 = *(_QWORD **)(v13 - 8);
      a5 = (__int64)&a6[v18];
    }
    else
    {
      a5 = v13;
      a6 = (_QWORD *)(v13 - v18 * 8);
    }
    for ( result = (unsigned int)v23; (_QWORD *)a5 != a6; LODWORD(v23) = v23 + 1 )
    {
      a4 = HIDWORD(v23);
      v19 = *a6;
      if ( result + 1 > (unsigned __int64)HIDWORD(v23) )
      {
        v10 = v24;
        v20 = a5;
        v21 = a6;
        sub_C8D5F0((__int64)&v22, v24, result + 1, 8u, a5, (__int64)a6);
        result = (unsigned int)v23;
        a5 = v20;
        a6 = v21;
      }
      a6 += 4;
      v22[result] = v19;
      result = (unsigned int)(v23 + 1);
    }
LABEL_16:
    v9 = v22;
  }
LABEL_9:
  if ( v9 != v24 )
    result = _libc_free(v9, v10);
  if ( !v29 )
    return _libc_free(v26, v10);
  return result;
}
