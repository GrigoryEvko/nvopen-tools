// Function: sub_103D3D0
// Address: 0x103d3d0
//
_BYTE *__fastcall sub_103D3D0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 *v5; // rbx
  char v6; // r14
  _BYTE *v7; // rax
  __int64 v8; // r12
  unsigned __int8 *v9; // r9
  _BYTE *v10; // rax
  __int64 v11; // rdi
  const char *v12; // rax
  size_t v13; // rdx
  _BYTE *v14; // rdi
  unsigned __int8 *v15; // rsi
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  void *v18; // rdx
  _BYTE *result; // rax
  __int64 v20; // rax
  unsigned __int8 *v21; // [rsp+0h] [rbp-40h]
  unsigned __int8 *v22; // [rsp+0h] [rbp-40h]
  size_t v23; // [rsp+0h] [rbp-40h]
  __int64 *v24; // [rsp+8h] [rbp-38h]

  v4 = sub_CB59D0(a2, *(unsigned int *)(a1 + 72));
  sub_904010(v4, " = MemoryPhi(");
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
  {
    v5 = *(__int64 **)(a1 - 8);
    v24 = &v5[4 * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)];
  }
  else
  {
    v24 = (__int64 *)a1;
    v5 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  }
  v6 = 1;
  if ( v5 != v24 )
  {
    while ( 1 )
    {
      v8 = *v5;
      v9 = *(unsigned __int8 **)(*(_QWORD *)(a1 - 8)
                               + 32LL * *(unsigned int *)(a1 + 76)
                               + 8LL * (unsigned int)(((__int64)v5 - *(_QWORD *)(a1 - 8)) >> 5));
      v10 = *(_BYTE **)(a2 + 32);
      if ( v6 )
        break;
      if ( v10 != *(_BYTE **)(a2 + 24) )
      {
        *v10 = 44;
        v11 = a2;
        v10 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 1LL);
        *(_QWORD *)(a2 + 32) = v10;
LABEL_10:
        if ( *(_QWORD *)(v11 + 24) > (unsigned __int64)v10 )
          goto LABEL_11;
        goto LABEL_30;
      }
      v21 = *(unsigned __int8 **)(*(_QWORD *)(a1 - 8)
                                + 32LL * *(unsigned int *)(a1 + 76)
                                + 8LL * (unsigned int)(((__int64)v5 - *(_QWORD *)(a1 - 8)) >> 5));
      v20 = sub_CB6200(a2, (unsigned __int8 *)",", 1u);
      v9 = v21;
      v11 = v20;
      v10 = *(_BYTE **)(v20 + 32);
      if ( *(_QWORD *)(v11 + 24) > (unsigned __int64)v10 )
      {
LABEL_11:
        *(_QWORD *)(v11 + 32) = v10 + 1;
        *v10 = 123;
        goto LABEL_12;
      }
LABEL_30:
      v22 = v9;
      sub_CB5D20(v11, 123);
      v9 = v22;
LABEL_12:
      if ( (v9[7] & 0x10) != 0 )
      {
        v12 = sub_BD5D20((__int64)v9);
        v14 = *(_BYTE **)(a2 + 32);
        v15 = (unsigned __int8 *)v12;
        v16 = *(_QWORD *)(a2 + 24);
        if ( v13 > v16 - (unsigned __int64)v14 )
        {
          sub_CB6200(a2, v15, v13);
          v14 = *(_BYTE **)(a2 + 32);
          v16 = *(_QWORD *)(a2 + 24);
        }
        else if ( v13 )
        {
          v23 = v13;
          memcpy(v14, v15, v13);
          v16 = *(_QWORD *)(a2 + 24);
          v14 = (_BYTE *)(v23 + *(_QWORD *)(a2 + 32));
          *(_QWORD *)(a2 + 32) = v14;
        }
        if ( v16 > (unsigned __int64)v14 )
        {
LABEL_17:
          *(_QWORD *)(a2 + 32) = v14 + 1;
          *v14 = 44;
          if ( *(_BYTE *)v8 == 27 )
            goto LABEL_18;
          goto LABEL_27;
        }
      }
      else
      {
        sub_A5BF40(v9, a2, 0, 0);
        v14 = *(_BYTE **)(a2 + 32);
        if ( *(_QWORD *)(a2 + 24) > (unsigned __int64)v14 )
          goto LABEL_17;
      }
      sub_CB5D20(a2, 44);
      if ( *(_BYTE *)v8 == 27 )
      {
LABEL_18:
        v17 = *(unsigned int *)(v8 + 80);
        goto LABEL_19;
      }
LABEL_27:
      v17 = *(unsigned int *)(v8 + 72);
LABEL_19:
      if ( (_DWORD)v17 )
      {
        sub_CB59D0(a2, v17);
        v7 = *(_BYTE **)(a2 + 32);
        goto LABEL_6;
      }
      v18 = *(void **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v18 <= 0xAu )
      {
        sub_CB6200(a2, "liveOnEntry", 0xBu);
        v7 = *(_BYTE **)(a2 + 32);
LABEL_6:
        if ( (unsigned __int64)v7 < *(_QWORD *)(a2 + 24) )
          goto LABEL_7;
LABEL_22:
        v5 += 4;
        sub_CB5D20(a2, 125);
        if ( v24 == v5 )
          goto LABEL_23;
      }
      else
      {
        qmemcpy(v18, "liveOnEntry", 11);
        v7 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 11LL);
        *(_QWORD *)(a2 + 32) = v7;
        if ( (unsigned __int64)v7 >= *(_QWORD *)(a2 + 24) )
          goto LABEL_22;
LABEL_7:
        v5 += 4;
        *(_QWORD *)(a2 + 32) = v7 + 1;
        *v7 = 125;
        if ( v24 == v5 )
          goto LABEL_23;
      }
    }
    v11 = a2;
    v6 = 0;
    goto LABEL_10;
  }
LABEL_23:
  result = *(_BYTE **)(a2 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(a2 + 24) )
    return (_BYTE *)sub_CB5D20(a2, 41);
  *(_QWORD *)(a2 + 32) = result + 1;
  *result = 41;
  return result;
}
