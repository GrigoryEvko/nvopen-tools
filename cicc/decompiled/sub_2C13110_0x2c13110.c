// Function: sub_2C13110
// Address: 0x2c13110
//
__int64 __fastcall sub_2C13110(__int64 a1, void *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // r9
  __int64 v8; // r8
  __int64 result; // rax
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // r14
  void *v17; // r15
  __int64 v18; // rbx
  _QWORD *v19; // rdi
  _QWORD *v20; // rax
  __int64 v21; // rdx
  unsigned int n; // [rsp+18h] [rbp-C8h]
  size_t nb; // [rsp+18h] [rbp-C8h]
  size_t na; // [rsp+18h] [rbp-C8h]
  size_t nc; // [rsp+18h] [rbp-C8h]
  size_t nd; // [rsp+18h] [rbp-C8h]
  __int64 v28; // [rsp+20h] [rbp-C0h]
  __int64 v29; // [rsp+28h] [rbp-B8h]
  _QWORD v30[4]; // [rsp+30h] [rbp-B0h] BYREF
  __int16 v31; // [rsp+50h] [rbp-90h]
  _QWORD *v32; // [rsp+60h] [rbp-80h] BYREF
  __int64 v33; // [rsp+68h] [rbp-78h]
  _QWORD v34[14]; // [rsp+70h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(*(_QWORD *)a2 + 8LL);
  if ( sub_BCEA30(v6) )
  {
    v8 = 8 * a3;
    v32 = v34;
    v33 = 0x600000000LL;
    if ( (unsigned __int64)(8 * a3) > 0x30 )
    {
      sub_C8D5F0((__int64)&v32, v34, (8 * a3) >> 3, 8u, v8, v7);
      v8 = 8 * a3;
      v19 = &v32[(unsigned int)v33];
    }
    else
    {
      if ( !v8 )
      {
        result = v34[0];
        LODWORD(v33) = (8 * a3) >> 3;
        n = (unsigned int)a3 >> 1;
        v10 = *(_QWORD *)(v34[0] + 8LL);
        if ( !((unsigned int)a3 >> 1) )
          return result;
        do
        {
LABEL_5:
          v11 = 0;
          BYTE4(v29) = *(_BYTE *)(v10 + 8) == 18;
          LODWORD(v29) = 2 * *(_DWORD *)(v10 + 32);
          v10 = sub_BCE1B0(*(__int64 **)(v10 + 24), v29);
          v12 = n;
          do
          {
            BYTE4(v28) = 0;
            v30[0] = v32[v11];
            v13 = v32[v12++];
            v30[1] = v13;
            v14 = sub_B35180(a1, v10, 0x17Fu, (__int64)v30, 2u, v28, a4);
            v32[v11++] = v14;
          }
          while ( n > (unsigned int)v11 );
          n >>= 1;
        }
        while ( n );
        v15 = v32;
        result = *v32;
LABEL_9:
        if ( v15 == v34 )
          return result;
        goto LABEL_10;
      }
      v19 = v34;
    }
    memcpy(v19, a2, v8);
    v15 = v32;
    LODWORD(v33) = ((8 * a3) >> 3) + v33;
    result = *v32;
    n = (unsigned int)a3 >> 1;
    v10 = *(_QWORD *)(*v32 + 8LL);
    if ( !((unsigned int)a3 >> 1) )
      goto LABEL_9;
    goto LABEL_5;
  }
  v16 = sub_9B9840((unsigned int **)a1, (char *)a2, a3);
  sub_9B9520((__int64)&v32, *(_DWORD *)(v6 + 32), a3);
  v17 = v32;
  v18 = (unsigned int)v33;
  na = sub_ACADE0(*(__int64 ***)(v16 + 8));
  result = (*(__int64 (__fastcall **)(_QWORD, __int64, size_t, void *, __int64))(**(_QWORD **)(a1 + 80) + 112LL))(
             *(_QWORD *)(a1 + 80),
             v16,
             na,
             v17,
             v18);
  if ( result )
  {
    v15 = v32;
    if ( v32 == v34 )
      return result;
LABEL_10:
    nb = result;
    _libc_free((unsigned __int64)v15);
    return nb;
  }
  v31 = 257;
  v20 = sub_BD2C40(112, unk_3F1FE60);
  if ( v20 )
  {
    v21 = na;
    nc = (size_t)v20;
    sub_B4E9E0((__int64)v20, v16, v21, v17, v18, (__int64)v30, 0, 0);
    v20 = (_QWORD *)nc;
  }
  nd = (size_t)v20;
  (*(void (__fastcall **)(_QWORD, _QWORD *, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v20,
    a4,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64));
  sub_94AAF0((unsigned int **)a1, nd);
  v15 = v32;
  result = nd;
  if ( v32 != v34 )
    goto LABEL_10;
  return result;
}
