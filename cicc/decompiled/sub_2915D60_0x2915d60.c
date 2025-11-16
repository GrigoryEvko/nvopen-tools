// Function: sub_2915D60
// Address: 0x2915d60
//
void __fastcall sub_2915D60(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r10
  unsigned int v8; // eax
  int v9; // r8d
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  int v13; // ecx
  unsigned __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned __int8 *v16; // rdi
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r15
  __int64 *v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  _QWORD *v28; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v29; // [rsp+18h] [rbp-98h]
  _QWORD v30[4]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v31; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v32; // [rsp+48h] [rbp-68h]
  __int64 v33; // [rsp+50h] [rbp-60h]
  int v34; // [rsp+58h] [rbp-58h]
  char v35; // [rsp+5Ch] [rbp-54h]
  __int64 v36; // [rsp+60h] [rbp-50h] BYREF

  v6 = v30;
  v32 = &v36;
  v33 = 0x100000004LL;
  v29 = 0x400000001LL;
  v8 = 1;
  v28 = v30;
  v34 = 0;
  v35 = 1;
  v36 = a2;
  v31 = 1;
  v30[0] = a2;
  do
  {
    while ( 1 )
    {
      v15 = v8--;
      v16 = (unsigned __int8 *)v6[v15 - 1];
      LODWORD(v29) = v8;
      v17 = *v16;
      if ( (_BYTE)v17 == 61 )
      {
        v9 = *((unsigned __int16 *)v16 + 1);
        _BitScanReverse64(&v10, 1LL << *(_WORD *)(a1[4] + 2LL));
        v11 = (0x8000000000000000LL >> ((unsigned __int8)v10 ^ 0x3Fu)) | (a1[14] - a1[5]);
        _BitScanReverse64(&v12, 1LL << (*((_WORD *)v16 + 1) >> 1));
        v13 = 63 - (v12 ^ 0x3F);
        v14 = -(__int64)v11 & v11;
        if ( v14 )
        {
          _BitScanReverse64(&v14, v14);
          if ( (unsigned __int8)v13 > (unsigned __int8)(63 - (v14 ^ 0x3F)) )
            v13 = 63 - (v14 ^ 0x3F);
        }
        goto LABEL_5;
      }
      if ( (_BYTE)v17 != 62 )
        break;
      v9 = *((unsigned __int16 *)v16 + 1);
      _BitScanReverse64(&v18, 1LL << *(_WORD *)(a1[4] + 2LL));
      v19 = (0x8000000000000000LL >> ((unsigned __int8)v18 ^ 0x3Fu)) | (a1[14] - a1[5]);
      _BitScanReverse64(&v20, 1LL << (*((_WORD *)v16 + 1) >> 1));
      v13 = 63 - (v20 ^ 0x3F);
      v21 = -(__int64)v19 & v19;
      if ( v21 )
      {
        _BitScanReverse64((unsigned __int64 *)&v22, v21);
        if ( (unsigned __int8)v13 > (unsigned __int8)(63 - (v22 ^ 0x3F)) )
          v13 = 63 - (v22 ^ 0x3F);
      }
LABEL_5:
      a4 = (unsigned int)(2 * v13);
      a5 = (unsigned int)a4 | v9 & 0xFFFFFF81;
      *((_WORD *)v16 + 1) = a5;
      v8 = v29;
      v6 = v28;
LABEL_6:
      if ( !v8 )
        goto LABEL_21;
    }
    v23 = *((_QWORD *)v16 + 2);
    if ( !v23 )
      goto LABEL_6;
    do
    {
      while ( 1 )
      {
        v24 = *(_QWORD *)(v23 + 24);
        if ( v35 )
        {
          v25 = v32;
          a4 = HIDWORD(v33);
          v17 = (__int64)&v32[HIDWORD(v33)];
          if ( v32 != (__int64 *)v17 )
          {
            while ( v24 != *v25 )
            {
              if ( (__int64 *)v17 == ++v25 )
                goto LABEL_30;
            }
            goto LABEL_19;
          }
LABEL_30:
          if ( HIDWORD(v33) < (unsigned int)v33 )
            break;
        }
        sub_C8CC70((__int64)&v31, *(_QWORD *)(v23 + 24), v17, a4, a5, a6);
        if ( (_BYTE)v17 )
          goto LABEL_26;
LABEL_19:
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          goto LABEL_20;
      }
      ++HIDWORD(v33);
      *(_QWORD *)v17 = v24;
      ++v31;
LABEL_26:
      v26 = (unsigned int)v29;
      a4 = HIDWORD(v29);
      v27 = (unsigned int)v29 + 1LL;
      if ( v27 > HIDWORD(v29) )
      {
        sub_C8D5F0((__int64)&v28, v30, v27, 8u, a5, a6);
        v26 = (unsigned int)v29;
      }
      v17 = (__int64)v28;
      v28[v26] = v24;
      LODWORD(v29) = v29 + 1;
      v23 = *(_QWORD *)(v23 + 8);
    }
    while ( v23 );
LABEL_20:
    v8 = v29;
    v6 = v28;
  }
  while ( (_DWORD)v29 );
LABEL_21:
  if ( v6 != v30 )
    _libc_free((unsigned __int64)v6);
  if ( !v35 )
    _libc_free((unsigned __int64)v32);
}
