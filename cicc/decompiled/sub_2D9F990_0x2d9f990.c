// Function: sub_2D9F990
// Address: 0x2d9f990
//
void __fastcall sub_2D9F990(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // r8
  __int64 v8; // r15
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rcx
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int64 v13; // r13
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // r15
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // r13
  unsigned __int64 v20; // rbx
  __int64 v21; // rax
  volatile signed __int32 *v22; // r15
  signed __int32 v23; // eax
  signed __int32 v24; // eax
  unsigned __int64 v25; // r13
  __int64 v26; // rax
  unsigned __int64 v27; // [rsp+8h] [rbp-48h]
  unsigned __int64 v28; // [rsp+10h] [rbp-40h]
  unsigned __int64 v29; // [rsp+18h] [rbp-38h]
  unsigned __int64 v30; // [rsp+18h] [rbp-38h]
  unsigned __int64 v31; // [rsp+18h] [rbp-38h]

  v3 = a1[1];
  if ( v3 != a1[2] )
  {
    if ( v3 )
    {
      *(_QWORD *)v3 = *(_QWORD *)a2;
      v4 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(v3 + 16) = 0;
      *(_QWORD *)(v3 + 8) = v4;
      v5 = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)(v3 + 16) = v5;
      LOBYTE(v5) = *(_BYTE *)(a2 + 32);
      v6 = *(_QWORD *)(a2 + 24);
      *(_QWORD *)(a2 + 16) = 0;
      *(_BYTE *)(v3 + 32) = v5;
      *(_QWORD *)(v3 + 24) = v6;
      v3 = a1[1];
    }
    a1[1] = v3 + 40;
    return;
  }
  v7 = *a1;
  v8 = v3 - *a1;
  v9 = 0xCCCCCCCCCCCCCCCDLL * (v8 >> 3);
  if ( v9 == 0x333333333333333LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v3 - *a1) >> 3);
  v11 = __CFADD__(v10, v9);
  v12 = v10 - 0x3333333333333333LL * ((__int64)(v3 - *a1) >> 3);
  if ( v11 )
  {
    v25 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_36:
    v31 = *a1;
    v26 = sub_22077B0(v25);
    v7 = v31;
    v14 = v26;
    v27 = v25 + v26;
    v13 = v26 + 40;
    goto LABEL_11;
  }
  if ( v12 )
  {
    if ( v12 > 0x333333333333333LL )
      v12 = 0x333333333333333LL;
    v25 = 40 * v12;
    goto LABEL_36;
  }
  v27 = 0;
  v13 = 40;
  v14 = 0;
LABEL_11:
  v15 = v14 + v8;
  if ( v15 )
  {
    v16 = *(_QWORD *)(a2 + 24);
    *(_QWORD *)v15 = *(_QWORD *)a2;
    v17 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(v15 + 24) = v16;
    *(_QWORD *)(v15 + 8) = v17;
    v18 = *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a2 + 8) = 0;
    *(_QWORD *)(v15 + 16) = v18;
    LOBYTE(v18) = *(_BYTE *)(a2 + 32);
    *(_QWORD *)(a2 + 16) = 0;
    *(_BYTE *)(v15 + 32) = v18;
  }
  if ( v3 != v7 )
  {
    v19 = v14;
    v20 = v7;
    while ( 1 )
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = *(_QWORD *)v20;
        *(_QWORD *)(v19 + 8) = *(_QWORD *)(v20 + 8);
        v21 = *(_QWORD *)(v20 + 16);
        *(_QWORD *)(v20 + 16) = 0;
        *(_QWORD *)(v19 + 16) = v21;
        LODWORD(v21) = *(_DWORD *)(v20 + 24);
        *(_QWORD *)(v20 + 8) = 0;
        *(_DWORD *)(v19 + 24) = v21;
        *(_DWORD *)(v19 + 28) = *(_DWORD *)(v20 + 28);
        *(_BYTE *)(v19 + 32) = *(_BYTE *)(v20 + 32);
      }
      v22 = *(volatile signed __int32 **)(v20 + 16);
      if ( v22 )
      {
        if ( &_pthread_key_create )
        {
          v23 = _InterlockedExchangeAdd(v22 + 2, 0xFFFFFFFF);
        }
        else
        {
          v23 = *((_DWORD *)v22 + 2);
          *((_DWORD *)v22 + 2) = v23 - 1;
        }
        if ( v23 == 1 )
        {
          v28 = v7;
          v29 = v14;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 16LL))(v22);
          v14 = v29;
          v7 = v28;
          if ( &_pthread_key_create )
          {
            v24 = _InterlockedExchangeAdd(v22 + 3, 0xFFFFFFFF);
          }
          else
          {
            v24 = *((_DWORD *)v22 + 3);
            *((_DWORD *)v22 + 3) = v24 - 1;
          }
          if ( v24 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 24LL))(v22);
            v7 = v28;
            v14 = v29;
          }
        }
      }
      v20 += 40LL;
      if ( v3 == v20 )
        break;
      v19 += 40LL;
    }
    v13 = v19 + 80;
  }
  if ( v7 )
  {
    v30 = v14;
    j_j___libc_free_0(v7);
    v14 = v30;
  }
  a1[1] = v13;
  *a1 = v14;
  a1[2] = v27;
}
