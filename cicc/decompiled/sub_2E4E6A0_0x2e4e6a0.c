// Function: sub_2E4E6A0
// Address: 0x2e4e6a0
//
__int64 __fastcall sub_2E4E6A0(__int64 a1, __int64 *a2)
{
  unsigned int v2; // r12d
  char v4; // al
  __int64 v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r14
  unsigned __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // r14
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r14
  unsigned __int64 v17; // rdi
  _QWORD v18[3]; // [rsp+0h] [rbp-120h] BYREF
  char v19; // [rsp+18h] [rbp-108h]
  __int64 v20; // [rsp+20h] [rbp-100h]
  __int64 v21; // [rsp+28h] [rbp-F8h]
  __int64 v22; // [rsp+30h] [rbp-F0h]
  __int64 v23; // [rsp+38h] [rbp-E8h]
  _BYTE *v24; // [rsp+40h] [rbp-E0h]
  __int64 v25; // [rsp+48h] [rbp-D8h]
  _BYTE v26[64]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v27; // [rsp+90h] [rbp-90h]
  __int64 v28; // [rsp+98h] [rbp-88h]
  __int64 v29; // [rsp+A0h] [rbp-80h]
  unsigned int v30; // [rsp+A8h] [rbp-78h]
  __int64 v31; // [rsp+B0h] [rbp-70h]
  __int64 v32; // [rsp+B8h] [rbp-68h]
  __int64 v33; // [rsp+C0h] [rbp-60h]
  unsigned int v34; // [rsp+C8h] [rbp-58h]
  __int64 v35; // [rsp+D0h] [rbp-50h]
  _QWORD *v36; // [rsp+D8h] [rbp-48h]
  __int64 v37; // [rsp+E0h] [rbp-40h]
  unsigned int v38; // [rsp+E8h] [rbp-38h]
  char v39; // [rsp+F0h] [rbp-30h]

  v2 = 0;
  if ( !(unsigned __int8)sub_BB98D0((_QWORD *)a1, *a2) )
  {
    v4 = *(_BYTE *)(a1 + 200);
    v39 = 0;
    v18[0] = 0;
    v18[1] = 0;
    if ( !v4 )
      v4 = qword_501F508;
    v18[2] = 0;
    v20 = 0;
    v19 = v4;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = v26;
    v25 = 0x800000000LL;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31 = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v2 = sub_2E4E590((__int64)v18, (__int64)a2);
    v5 = v38;
    if ( v38 )
    {
      v6 = v36;
      v7 = &v36[10 * v38];
      do
      {
        if ( *v6 != -8192 && *v6 != -4096 )
        {
          v8 = v6[1];
          if ( (_QWORD *)v8 != v6 + 3 )
            _libc_free(v8);
        }
        v6 += 10;
      }
      while ( v7 != v6 );
      v5 = v38;
    }
    sub_C7D6A0((__int64)v36, 80 * v5, 8);
    v9 = v34;
    if ( v34 )
    {
      v10 = v32;
      v11 = v32 + ((unsigned __int64)v34 << 7);
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v10 <= 0xFFFFFFFD )
          {
            v12 = *(_QWORD *)(v10 + 88);
            if ( v12 != v10 + 104 )
              _libc_free(v12);
            if ( !*(_BYTE *)(v10 + 52) )
              break;
          }
          v10 += 128;
          if ( v11 == v10 )
            goto LABEL_21;
        }
        v13 = *(_QWORD *)(v10 + 32);
        v10 += 128;
        _libc_free(v13);
      }
      while ( v11 != v10 );
LABEL_21:
      v9 = v34;
    }
    sub_C7D6A0(v32, v9 << 7, 8);
    v14 = v30;
    if ( v30 )
    {
      v15 = v28;
      v16 = v28 + 56LL * v30;
      do
      {
        while ( *(_QWORD *)v15 == -4096 || *(_QWORD *)v15 == -8192 || *(_BYTE *)(v15 + 36) )
        {
          v15 += 56;
          if ( v16 == v15 )
            goto LABEL_29;
        }
        v17 = *(_QWORD *)(v15 + 16);
        v15 += 56;
        _libc_free(v17);
      }
      while ( v16 != v15 );
LABEL_29:
      v14 = v30;
    }
    sub_C7D6A0(v28, 56 * v14, 8);
    if ( v24 != v26 )
      _libc_free((unsigned __int64)v24);
    sub_C7D6A0(v21, 8LL * (unsigned int)v23, 8);
  }
  return v2;
}
