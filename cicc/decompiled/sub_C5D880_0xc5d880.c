// Function: sub_C5D880
// Address: 0xc5d880
//
__int64 __fastcall sub_C5D880(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, char a7)
{
  unsigned int v7; // r10d
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rax
  _QWORD *v17; // rsi
  __int64 v18; // r8
  __int64 v19; // rcx
  __int64 v20; // rax
  int v21; // edi
  __int64 v22; // r8
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned int v28; // [rsp+8h] [rbp-48h]
  _QWORD *v29; // [rsp+8h] [rbp-48h]
  unsigned int v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+10h] [rbp-40h] BYREF
  int *v32[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = a2;
  if ( !a7 )
  {
    v29 = sub_C52410();
    v14 = sub_C959E0();
    v7 = a2;
    v31 = v14;
    v15 = v14;
    v16 = (_QWORD *)v29[2];
    v17 = v29 + 1;
    if ( !v16 )
      goto LABEL_21;
    do
    {
      while ( 1 )
      {
        v18 = v16[2];
        v19 = v16[3];
        if ( v15 <= v16[4] )
          break;
        v16 = (_QWORD *)v16[3];
        if ( !v19 )
          goto LABEL_10;
      }
      v17 = v16;
      v16 = (_QWORD *)v16[2];
    }
    while ( v18 );
LABEL_10:
    if ( v29 + 1 == v17 || v15 < v17[4] )
    {
LABEL_21:
      v32[0] = (int *)&v31;
      v26 = sub_C5D700(v29, v17, (unsigned __int64 **)v32);
      v7 = a2;
      v17 = (_QWORD *)v26;
    }
    v20 = v17[7];
    if ( v20 )
    {
      v21 = *(_DWORD *)(a1 + 8);
      v22 = (__int64)(v17 + 6);
      do
      {
        while ( 1 )
        {
          v23 = *(_QWORD *)(v20 + 16);
          v24 = *(_QWORD *)(v20 + 24);
          if ( *(_DWORD *)(v20 + 32) >= v21 )
            break;
          v20 = *(_QWORD *)(v20 + 24);
          if ( !v24 )
            goto LABEL_17;
        }
        v22 = v20;
        v20 = *(_QWORD *)(v20 + 16);
      }
      while ( v23 );
LABEL_17:
      if ( v17 + 6 != (_QWORD *)v22 && v21 >= *(_DWORD *)(v22 + 32) )
        goto LABEL_20;
    }
    else
    {
      v22 = (__int64)(v17 + 6);
    }
    v30 = v7;
    v32[0] = (int *)(a1 + 8);
    v25 = sub_C5D7D0(v17 + 5, v22, v32);
    v7 = v30;
    v22 = v25;
LABEL_20:
    ++*(_DWORD *)(v22 + 36);
  }
  if ( qword_4F83C98 )
  {
    v28 = v7;
    (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, __int64, __int64, __int64, __int64))(*(_QWORD *)qword_4F83C98
                                                                                                 + 16LL))(
      qword_4F83C98,
      a1,
      *(unsigned __int16 *)(a1 + 14),
      v7,
      a3,
      a4,
      a5,
      a6);
    v7 = v28;
  }
  return (**(__int64 (__fastcall ***)(__int64, _QWORD, __int64, __int64, __int64, __int64))a1)(a1, v7, a3, a4, a5, a6);
}
