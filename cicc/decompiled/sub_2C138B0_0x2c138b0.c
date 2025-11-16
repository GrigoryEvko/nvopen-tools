// Function: sub_2C138B0
// Address: 0x2c138b0
//
__int64 __fastcall sub_2C138B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 result; // rax
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rdi
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int64 v14; // rdx
  int v15; // r13d
  __int64 *v16; // rsi
  __int64 *v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  void *v20; // rcx
  __int64 *v21; // rax
  __int64 *v22; // rcx
  __int64 v23; // [rsp+8h] [rbp-F8h]
  __int64 v24; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v25; // [rsp+18h] [rbp-E8h]
  __int64 v26; // [rsp+18h] [rbp-E8h]
  __int64 v27; // [rsp+18h] [rbp-E8h]
  _QWORD *v28; // [rsp+18h] [rbp-E8h]
  __int64 v29; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v30; // [rsp+18h] [rbp-E8h]
  const char *v31; // [rsp+20h] [rbp-E0h] BYREF
  char v32; // [rsp+40h] [rbp-C0h]
  char v33; // [rsp+41h] [rbp-BFh]
  unsigned int v34[8]; // [rsp+50h] [rbp-B0h] BYREF
  __int16 v35; // [rsp+70h] [rbp-90h]
  __int64 *v36; // [rsp+80h] [rbp-80h] BYREF
  __int64 v37; // [rsp+88h] [rbp-78h]
  _QWORD v38[2]; // [rsp+90h] [rbp-70h] BYREF
  __int16 v39; // [rsp+A0h] [rbp-60h]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = **(_QWORD **)a1;
  if ( *(_BYTE *)(v4 + 12) )
  {
    v11 = sub_2BFB640(v4, v5, 0);
    v14 = **(unsigned int **)(a1 + 16);
    v36 = v38;
    v37 = 0x600000000LL;
    v15 = v14;
    if ( (unsigned int)v14 > 6 )
    {
      v30 = v14;
      sub_C8D5F0((__int64)&v36, v38, v14, 8u, v12, v13);
      v21 = v36;
      v14 = v30;
      v22 = &v36[v30];
      do
        *v21++ = v11;
      while ( v22 != v21 );
      LODWORD(v37) = v15;
      v16 = v36;
    }
    else
    {
      v16 = v38;
      if ( v14 )
      {
        v17 = v38;
        do
          *v17++ = v11;
        while ( &v38[v14] != v17 );
        v16 = v36;
      }
      LODWORD(v37) = v14;
    }
    *(_QWORD *)v34 = "interleaved.mask";
    v18 = *(_QWORD *)(a1 + 8);
    v35 = 259;
    result = sub_2C13110(*(_QWORD *)(v18 + 904), v16, v14, (__int64)v34);
    if ( v36 != v38 )
    {
      v27 = result;
      _libc_free((unsigned __int64)v36);
      return v27;
    }
  }
  else
  {
    result = a2;
    if ( v5 )
    {
      v33 = 1;
      v7 = sub_2BFB640(v4, v5, 0);
      v8 = *(_QWORD *)(a1 + 8);
      v32 = 3;
      v31 = "interleaved.mask";
      v9 = *(_QWORD *)(v8 + 904);
      sub_9B9470((__int64)&v36, **(_DWORD **)(a1 + 16), *(_DWORD *)(v8 + 8));
      v24 = (unsigned int)v37;
      v25 = (unsigned __int64)v36;
      v23 = sub_ACADE0(*(__int64 ***)(v7 + 8));
      result = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, unsigned __int64, __int64))(**(_QWORD **)(v9 + 80)
                                                                                              + 112LL))(
                 *(_QWORD *)(v9 + 80),
                 v7,
                 v23,
                 v25,
                 v24);
      if ( !result )
      {
        v35 = 257;
        v19 = sub_BD2C40(112, unk_3F1FE60);
        if ( v19 )
        {
          v20 = (void *)v25;
          v28 = v19;
          sub_B4E9E0((__int64)v19, v7, v23, v20, v24, (__int64)v34, 0, 0);
          v19 = v28;
        }
        v29 = (__int64)v19;
        (*(void (__fastcall **)(_QWORD, _QWORD *, const char **, _QWORD, _QWORD))(**(_QWORD **)(v9 + 88) + 16LL))(
          *(_QWORD *)(v9 + 88),
          v19,
          &v31,
          *(_QWORD *)(v9 + 56),
          *(_QWORD *)(v9 + 64));
        sub_94AAF0((unsigned int **)v9, v29);
        result = v29;
      }
      if ( v36 != v38 )
      {
        v26 = result;
        _libc_free((unsigned __int64)v36);
        result = v26;
      }
      if ( a2 )
      {
        v10 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 904LL);
        v34[1] = 0;
        v39 = 257;
        return sub_2C137C0(v10, 28, result, a2, v34[0], (__int64)&v36, 0);
      }
    }
  }
  return result;
}
