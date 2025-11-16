// Function: sub_2357470
// Address: 0x2357470
//
__int64 __fastcall sub_2357470(unsigned __int64 *a1, char *a2)
{
  char v2; // al
  void (__fastcall *v3)(_BYTE *, char *, __int64); // rax
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rbx
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // r8
  __int64 v13; // r12
  __int64 v14; // rbx
  _QWORD *v15; // rdi
  __int64 result; // rax
  _QWORD *v17; // [rsp+8h] [rbp-58h] BYREF
  char v18; // [rsp+10h] [rbp-50h]
  _BYTE v19[16]; // [rsp+18h] [rbp-48h] BYREF
  void (__fastcall *v20)(__int64, _BYTE *, __int64); // [rsp+28h] [rbp-38h]
  __int64 v21; // [rsp+30h] [rbp-30h]
  unsigned __int64 v22; // [rsp+38h] [rbp-28h]
  __int64 v23; // [rsp+40h] [rbp-20h]
  __int64 v24; // [rsp+48h] [rbp-18h]

  v2 = *a2;
  v20 = 0;
  v18 = v2;
  v3 = (void (__fastcall *)(_BYTE *, char *, __int64))*((_QWORD *)a2 + 3);
  if ( v3 )
  {
    v3(v19, a2 + 8, 2);
    v21 = *((_QWORD *)a2 + 4);
    v20 = (void (__fastcall *)(__int64, _BYTE *, __int64))*((_QWORD *)a2 + 3);
  }
  v4 = *((_QWORD *)a2 + 5);
  *((_QWORD *)a2 + 5) = 0;
  v22 = v4;
  v5 = *((_QWORD *)a2 + 6);
  *((_QWORD *)a2 + 6) = 0;
  v23 = v5;
  v6 = *((_QWORD *)a2 + 7);
  *((_DWORD *)a2 + 14) = 0;
  v24 = v6;
  v7 = sub_22077B0(0x48u);
  v8 = (_QWORD *)v7;
  if ( v7 )
  {
    *(_QWORD *)(v7 + 32) = 0;
    *(_QWORD *)v7 = &unk_4A0E878;
    *(_BYTE *)(v7 + 8) = v18;
    if ( v20 )
    {
      v20(v7 + 16, v19, 2);
      v8[5] = v21;
      v8[4] = v20;
    }
    v9 = v22;
    v22 = 0;
    v8[6] = v9;
    v10 = v23;
    v23 = 0;
    v8[7] = v10;
    v11 = v24;
    LODWORD(v24) = 0;
    v8[8] = v11;
  }
  v17 = v8;
  sub_2356EF0(a1, (unsigned __int64 *)&v17);
  if ( v17 )
    (*(void (__fastcall **)(_QWORD *))(*v17 + 8LL))(v17);
  v12 = v22;
  if ( HIDWORD(v23) && (_DWORD)v23 )
  {
    v13 = 8LL * (unsigned int)v23;
    v14 = 0;
    do
    {
      v15 = *(_QWORD **)(v12 + v14);
      if ( v15 != (_QWORD *)-8LL && v15 )
      {
        sub_C7D6A0((__int64)v15, *v15 + 9LL, 8);
        v12 = v22;
      }
      v14 += 8;
    }
    while ( v13 != v14 );
  }
  _libc_free(v12);
  result = (__int64)v20;
  if ( v20 )
    return ((__int64 (__fastcall *)(_BYTE *, _BYTE *, __int64))v20)(v19, v19, 3);
  return result;
}
