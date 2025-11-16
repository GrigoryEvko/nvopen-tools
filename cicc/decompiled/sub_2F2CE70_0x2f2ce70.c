// Function: sub_2F2CE70
// Address: 0x2f2ce70
//
__int64 __fastcall sub_2F2CE70(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  _BYTE *v5; // rax
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rax
  unsigned __int8 v8; // r15
  _QWORD *v9; // r15
  __int64 v10; // rsi
  _BYTE *v11; // rdx
  unsigned __int64 v12; // rdi
  _BYTE *v13; // rdx
  unsigned __int64 v14; // rdi
  unsigned __int8 v16; // [rsp+Fh] [rbp-121h]
  int v17; // [rsp+18h] [rbp-118h]
  _BYTE *v18; // [rsp+18h] [rbp-118h]
  _BYTE *v19; // [rsp+18h] [rbp-118h]
  __int64 v20; // [rsp+20h] [rbp-110h] BYREF
  __int64 v21; // [rsp+28h] [rbp-108h] BYREF
  __int64 v22; // [rsp+30h] [rbp-100h] BYREF
  __int64 v23; // [rsp+38h] [rbp-F8h]
  _QWORD *v24; // [rsp+40h] [rbp-F0h] BYREF
  unsigned int v25; // [rsp+48h] [rbp-E8h]
  _BYTE v26[48]; // [rsp+100h] [rbp-30h] BYREF

  v16 = 0;
  v20 = 0;
  v21 = 0;
  while ( (*(unsigned __int8 (__fastcall **)(__int64, __int64 *, __int64 *))(*(_QWORD *)a2 + 16LL))(a2, &v21, &v20) )
  {
    if ( (unsigned int)(v20 - 1) <= 0x3FFFFFFE )
      continue;
    v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 56LL) + 16 * (v20 & 0x7FFFFFFF));
    v5 = &v24;
    v22 = 0;
    v23 = 1;
    v6 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    do
    {
      *(_DWORD *)v5 = -1;
      v5 += 48;
      *((_DWORD *)v5 - 11) = -1;
    }
    while ( v5 != v26 );
    if ( (unsigned __int8)sub_2F2B9E0(a1, v6, HIDWORD(v20), v21, (__int64)&v22) )
    {
      v7 = sub_2F2A790(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v21, (__int64)&v22, 0);
      if ( (_DWORD)v7 )
      {
        v17 = v7;
        v8 = (*(__int64 (__fastcall **)(__int64, _QWORD, unsigned __int64))(*(_QWORD *)a2 + 24LL))(
               a2,
               (unsigned int)v7,
               HIDWORD(v7));
        if ( v8 )
        {
          sub_2EBF120(*(_QWORD *)(a1 + 24), v17);
          v16 = v8;
        }
        if ( (v23 & 1) != 0 )
        {
          v11 = v26;
          v9 = &v24;
          goto LABEL_18;
        }
        v9 = v24;
        v10 = 6LL * v25;
        if ( !v25 || (v11 = &v24[v10], &v24[v10] == v24) )
        {
LABEL_23:
          sub_C7D6A0((__int64)v9, v10 * 8, 8);
          continue;
        }
LABEL_18:
        while ( 2 )
        {
          while ( *(_DWORD *)v9 == -1 )
          {
            if ( *((_DWORD *)v9 + 1) != -1 )
              goto LABEL_15;
            v9 += 6;
            if ( v11 == (_BYTE *)v9 )
              goto LABEL_21;
          }
          if ( *(_DWORD *)v9 != -2 || *((_DWORD *)v9 + 1) != -2 )
          {
LABEL_15:
            v12 = v9[1];
            if ( (_QWORD *)v12 != v9 + 3 )
            {
              v18 = v11;
              _libc_free(v12);
              v11 = v18;
            }
          }
          v9 += 6;
          if ( v11 == (_BYTE *)v9 )
            goto LABEL_21;
          continue;
        }
      }
    }
    if ( (v23 & 1) != 0 )
    {
      v13 = v26;
      v9 = &v24;
      goto LABEL_32;
    }
    v9 = v24;
    v10 = 6LL * v25;
    if ( !v25 )
      goto LABEL_23;
    v13 = &v24[v10];
    if ( v24 == &v24[v10] )
      goto LABEL_23;
    do
    {
LABEL_32:
      if ( *(_DWORD *)v9 == -1 )
      {
        if ( *((_DWORD *)v9 + 1) == -1 )
          goto LABEL_31;
      }
      else if ( *(_DWORD *)v9 == -2 && *((_DWORD *)v9 + 1) == -2 )
      {
        goto LABEL_31;
      }
      v14 = v9[1];
      if ( (_QWORD *)v14 != v9 + 3 )
      {
        v19 = v13;
        _libc_free(v14);
        v13 = v19;
      }
LABEL_31:
      v9 += 6;
    }
    while ( v13 != (_BYTE *)v9 );
LABEL_21:
    if ( (v23 & 1) == 0 )
    {
      v9 = v24;
      v10 = 6LL * v25;
      goto LABEL_23;
    }
  }
  return v16;
}
