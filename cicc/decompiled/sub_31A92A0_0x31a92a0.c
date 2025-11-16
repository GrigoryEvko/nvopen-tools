// Function: sub_31A92A0
// Address: 0x31a92a0
//
__int64 __fastcall sub_31A92A0(__int64 a1, __int64 a2, char a3)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  _BYTE *v7; // rbx
  _BYTE *v8; // r14
  __int64 v9; // rax
  _DWORD *v10; // rax
  unsigned int v11; // r13d
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  _BYTE *v17; // [rsp+8h] [rbp-848h]
  __int64 v19; // [rsp+30h] [rbp-820h]
  _BYTE *v20; // [rsp+38h] [rbp-818h] BYREF
  __int64 v21; // [rsp+40h] [rbp-810h]
  _BYTE v22[128]; // [rsp+48h] [rbp-808h] BYREF
  _QWORD *v23; // [rsp+C8h] [rbp-788h] BYREF
  _QWORD v24[2]; // [rsp+D8h] [rbp-778h] BYREF
  _QWORD *v25; // [rsp+E8h] [rbp-768h] BYREF
  _QWORD v26[2]; // [rsp+F8h] [rbp-758h] BYREF
  int v27; // [rsp+108h] [rbp-748h]
  _BYTE *v28; // [rsp+110h] [rbp-740h] BYREF
  __int64 v29; // [rsp+118h] [rbp-738h]
  _BYTE v30[1840]; // [rsp+120h] [rbp-730h] BYREF

  v28 = v30;
  v29 = 0x800000000LL;
  sub_D39570(a1, (unsigned int *)&v28);
  v6 = (unsigned int)v29;
  v7 = v28;
  v17 = &v28[224 * (unsigned int)v29];
  if ( v17 == v28 )
  {
    v12 = (unsigned __int64)v28;
LABEL_27:
    v11 = 0;
  }
  else
  {
    v8 = v28;
    while ( 1 )
    {
      v9 = *(_QWORD *)v8;
      v20 = v22;
      v19 = v9;
      v21 = 0x800000000LL;
      if ( *((_DWORD *)v8 + 4) )
        sub_31A3D20((__int64)&v20, (__int64)(v8 + 8), v6, v3, v4, v5);
      v23 = v24;
      sub_31A4020((__int64 *)&v23, *((_BYTE **)v8 + 19), *((_QWORD *)v8 + 19) + *((_QWORD *)v8 + 20));
      v25 = v26;
      sub_31A4020((__int64 *)&v25, *((_BYTE **)v8 + 23), *((_QWORD *)v8 + 23) + *((_QWORD *)v8 + 24));
      v27 = *((_DWORD *)v8 + 54);
      if ( !a3 || (_DWORD)v19 == (_DWORD)a2 && BYTE4(v19) == BYTE4(a2) )
      {
        v6 = (unsigned int)v21;
        if ( (_DWORD)v21 )
          break;
      }
LABEL_5:
      if ( v25 != v26 )
        j_j___libc_free_0((unsigned __int64)v25);
      if ( v23 != v24 )
        j_j___libc_free_0((unsigned __int64)v23);
      if ( v20 != v22 )
        _libc_free((unsigned __int64)v20);
      v8 += 224;
      if ( v17 == v8 )
      {
        v7 = v28;
        v12 = (unsigned __int64)&v28[224 * (unsigned int)v29];
        goto LABEL_27;
      }
    }
    v10 = v20 + 4;
    v6 = (__int64)&v20[16 * (unsigned int)(v21 - 1) + 20];
    while ( *v10 != 10 )
    {
      v10 += 4;
      if ( (_DWORD *)v6 == v10 )
        goto LABEL_5;
    }
    if ( v25 != v26 )
      j_j___libc_free_0((unsigned __int64)v25);
    if ( v23 != v24 )
      j_j___libc_free_0((unsigned __int64)v23);
    if ( v20 != v22 )
      _libc_free((unsigned __int64)v20);
    v7 = v28;
    v11 = 1;
    v12 = (unsigned __int64)&v28[224 * (unsigned int)v29];
  }
  if ( v7 != (_BYTE *)v12 )
  {
    do
    {
      v12 -= 224LL;
      v13 = *(_QWORD *)(v12 + 184);
      if ( v13 != v12 + 200 )
        j_j___libc_free_0(v13);
      v14 = *(_QWORD *)(v12 + 152);
      if ( v14 != v12 + 168 )
        j_j___libc_free_0(v14);
      v15 = *(_QWORD *)(v12 + 8);
      if ( v15 != v12 + 24 )
        _libc_free(v15);
    }
    while ( (_BYTE *)v12 != v7 );
    v12 = (unsigned __int64)v28;
  }
  if ( (_BYTE *)v12 != v30 )
    _libc_free(v12);
  return v11;
}
