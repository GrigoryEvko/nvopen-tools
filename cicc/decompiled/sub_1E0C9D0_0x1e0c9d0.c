// Function: sub_1E0C9D0
// Address: 0x1e0c9d0
//
__int64 __fastcall sub_1E0C9D0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, int a6)
{
  __int64 v6; // r15
  _QWORD *v7; // rax
  unsigned __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v11; // rdx
  __int64 v12; // [rsp+10h] [rbp-B0h] BYREF
  char *v13; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v14; // [rsp+20h] [rbp-A0h]
  char v15; // [rsp+28h] [rbp-98h] BYREF
  char *v16; // [rsp+30h] [rbp-90h] BYREF
  __int64 v17; // [rsp+38h] [rbp-88h]
  char v18; // [rsp+40h] [rbp-80h] BYREF
  char *v19; // [rsp+48h] [rbp-78h] BYREF
  __int64 v20; // [rsp+50h] [rbp-70h]
  _BYTE v21[16]; // [rsp+58h] [rbp-68h] BYREF
  __int64 v22; // [rsp+68h] [rbp-58h]
  __int64 v23; // [rsp+70h] [rbp-50h]
  __int64 v24; // [rsp+78h] [rbp-48h]
  __int64 v25; // [rsp+80h] [rbp-40h]

  v6 = a1[52];
  v7 = (_QWORD *)a1[51];
  v8 = 0xEEEEEEEEEEEEEEEFLL * ((v6 - (__int64)v7) >> 3);
  if ( (_DWORD)v8 )
  {
    v9 = (__int64)&v7[15 * (unsigned int)(v8 - 1) + 15];
    while ( 1 )
    {
      a5 = v7;
      if ( *v7 == a2 )
        break;
      v7 += 15;
      if ( (_QWORD *)v9 == v7 )
        goto LABEL_6;
    }
  }
  else
  {
LABEL_6:
    v12 = a2;
    v14 = 0x100000000LL;
    v13 = &v15;
    v16 = &v18;
    v17 = 0x100000000LL;
    v19 = v21;
    v20 = 0x100000000LL;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    if ( v6 == a1[53] )
    {
      sub_1E0C430((__int64)(a1 + 51), (char *)v6, (unsigned __int64)&v12, (__int64)v21, (int)a5, a6);
      if ( v23 )
        j_j___libc_free_0(v23, v25 - v23);
    }
    else
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = a2;
        *(_QWORD *)(v6 + 8) = v6 + 24;
        *(_QWORD *)(v6 + 16) = 0x100000000LL;
        if ( (_DWORD)v14 )
          sub_1E09880(v6 + 8, &v13, v6 + 24, (__int64)v21, (int)a5, a6);
        *(_QWORD *)(v6 + 32) = v6 + 48;
        *(_QWORD *)(v6 + 40) = 0x100000000LL;
        v11 = (unsigned int)v17;
        if ( (_DWORD)v17 )
          sub_1E09880(v6 + 32, &v16, (unsigned int)v17, (__int64)v21, (int)a5, a6);
        *(_QWORD *)(v6 + 56) = v6 + 72;
        *(_QWORD *)(v6 + 64) = 0x100000000LL;
        if ( (_DWORD)v20 )
          sub_1E096E0(v6 + 56, &v19, v11, (__int64)v21, (int)a5, a6);
        *(_QWORD *)(v6 + 88) = v22;
        *(_QWORD *)(v6 + 96) = v23;
        *(_QWORD *)(v6 + 104) = v24;
        *(_QWORD *)(v6 + 112) = v25;
        v6 = a1[52];
      }
      a1[52] = v6 + 120;
    }
    if ( v19 != v21 )
      _libc_free((unsigned __int64)v19);
    if ( v16 != &v18 )
      _libc_free((unsigned __int64)v16);
    if ( v13 != &v15 )
      _libc_free((unsigned __int64)v13);
    return a1[51] + 120LL * (unsigned int)v8;
  }
  return (__int64)a5;
}
