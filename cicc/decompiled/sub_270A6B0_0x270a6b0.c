// Function: sub_270A6B0
// Address: 0x270a6b0
//
__int64 __fastcall sub_270A6B0(__int64 a1, __int64 a2, __int64 a3)
{
  void *v4; // rsi
  __int64 v5; // r13
  _BYTE *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r13
  bool v14; // r14
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r9
  _QWORD *v18; // r10
  __int64 v19; // rsi
  unsigned __int8 v20; // al
  _QWORD *v21; // r13
  __int64 v22; // rdi
  char v23; // al
  int v24; // eax
  _QWORD *v25; // rsi
  bool v26; // si
  char v27; // [rsp+17h] [rbp-A9h]
  _QWORD *v28; // [rsp+18h] [rbp-A8h]
  _QWORD *v29; // [rsp+18h] [rbp-A8h]
  __int64 v30; // [rsp+20h] [rbp-A0h]
  __int64 v31; // [rsp+20h] [rbp-A0h]
  __int64 v32; // [rsp+28h] [rbp-98h]
  __int64 v33; // [rsp+28h] [rbp-98h]
  __int64 v34; // [rsp+30h] [rbp-90h] BYREF
  _QWORD *v35; // [rsp+38h] [rbp-88h]
  int v36; // [rsp+40h] [rbp-80h]
  int v37; // [rsp+44h] [rbp-7Ch]
  int v38; // [rsp+48h] [rbp-78h]
  char v39; // [rsp+4Ch] [rbp-74h]
  _QWORD v40[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v41; // [rsp+60h] [rbp-60h] BYREF
  _BYTE *v42; // [rsp+68h] [rbp-58h]
  __int64 v43; // [rsp+70h] [rbp-50h]
  int v44; // [rsp+78h] [rbp-48h]
  char v45; // [rsp+7Ch] [rbp-44h]
  _BYTE v46[64]; // [rsp+80h] [rbp-40h] BYREF

  if ( !unk_5031DC8
    || !sub_270A460(a3)
    || (v8 = sub_BA8CD0(a3, (__int64)"llvm.global_ctors", 0x11u, 0)) == 0
    || (v9 = *((_QWORD *)v8 - 4), v10 = 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF), v11 = v9 - v10, v9 == v9 - v10) )
  {
    v4 = (void *)(a1 + 32);
    v5 = a1 + 80;
LABEL_3:
    *(_QWORD *)(a1 + 8) = v4;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v5;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v27 = 0;
  do
  {
    while ( 1 )
    {
      v12 = 32 * (1LL - (*(_DWORD *)(*(_QWORD *)v11 + 4LL) & 0x7FFFFFF));
      v13 = *(_QWORD *)(*(_QWORD *)v11 + v12);
      if ( !*(_BYTE *)v13 )
      {
        v14 = sub_B2FC80(*(_QWORD *)(*(_QWORD *)v11 + v12));
        if ( !v14 )
        {
          v15 = *(_QWORD *)(v13 + 80);
          if ( *(_QWORD *)(v15 + 8) == v13 + 72 )
          {
            v16 = *(_QWORD *)(v15 + 32);
            v17 = v15 + 24;
            v18 = 0;
            if ( v15 + 24 != v16 )
              break;
          }
        }
      }
      v11 += 32;
      if ( v9 == v11 )
        goto LABEL_26;
    }
    do
    {
      while ( 1 )
      {
        v19 = v16;
        v16 = *(_QWORD *)(v16 + 8);
        v20 = *(_BYTE *)(v19 - 24);
        if ( v20 <= 0x1Cu )
          goto LABEL_16;
        v21 = (_QWORD *)(v19 - 24);
        if ( v20 != 85 )
        {
          if ( v20 == 34 )
            break;
          goto LABEL_16;
        }
        v22 = *(_QWORD *)(v19 - 56);
        if ( !v22 || *(_BYTE *)v22 || *(_QWORD *)(v22 + 24) != *(_QWORD *)(v19 + 56) )
          break;
        v29 = v18;
        v31 = v16;
        v33 = v17;
        v24 = sub_3108960();
        v17 = v33;
        v16 = v31;
        v18 = v29;
        switch ( v24 )
        {
          case 8:
            if ( v29 )
            {
              v25 = *(_QWORD **)(v19 - 32LL * (*(_DWORD *)(v19 - 20) & 0x7FFFFFF) - 24);
              v26 = v25 != 0 && v25 == v29;
              if ( v26 )
              {
                sub_B43D60(v21);
                sub_B43D60(v29);
                v17 = v33;
                v18 = 0;
                v16 = v31;
                v14 = v26;
              }
              else
              {
                v18 = 0;
              }
            }
            break;
          case 21:
            goto LABEL_22;
          case 7:
            v18 = (_QWORD *)(v19 - 24);
            break;
        }
LABEL_16:
        if ( v17 == v16 )
          goto LABEL_25;
      }
LABEL_22:
      v28 = v18;
      v30 = v16;
      v32 = v17;
      v23 = sub_270A340((__int64)v21, 0);
      v18 = v28;
      v17 = v32;
      v16 = v30;
      if ( v23 )
        v18 = 0;
    }
    while ( v32 != v30 );
LABEL_25:
    v11 += 32;
    v27 |= v14;
  }
  while ( v9 != v11 );
LABEL_26:
  v4 = (void *)(a1 + 32);
  v5 = a1 + 80;
  if ( !v27 )
    goto LABEL_3;
  v35 = v40;
  v36 = 2;
  v40[0] = &unk_4F82408;
  v38 = 0;
  v39 = 1;
  v41 = 0;
  v42 = v46;
  v43 = 2;
  v44 = 0;
  v45 = 1;
  v37 = 1;
  v34 = 1;
  sub_C8CF70(a1, v4, 2, (__int64)v40, (__int64)&v34);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v46, (__int64)&v41);
  if ( !v45 )
    _libc_free((unsigned __int64)v42);
  if ( !v39 )
    _libc_free((unsigned __int64)v35);
  return a1;
}
