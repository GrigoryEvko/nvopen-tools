// Function: sub_354E8C0
// Address: 0x354e8c0
//
__int64 __fastcall sub_354E8C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r12
  char *v13; // rbx
  char *v14; // r12
  unsigned __int64 v15; // r14
  volatile signed __int32 *v16; // r15
  signed __int32 v17; // edx
  volatile signed __int32 *v18; // r15
  signed __int32 v19; // edx
  signed __int32 v21; // eax
  signed __int32 v22; // eax
  unsigned int v23; // [rsp+Ch] [rbp-224h]
  _BYTE v24[48]; // [rsp+10h] [rbp-220h] BYREF
  char *v25; // [rsp+40h] [rbp-1F0h]
  int v26; // [rsp+48h] [rbp-1E8h]
  char v27; // [rsp+50h] [rbp-1E0h] BYREF
  unsigned __int64 *v28; // [rsp+80h] [rbp-1B0h]
  unsigned int v29; // [rsp+88h] [rbp-1A8h]
  char v30; // [rsp+90h] [rbp-1A0h] BYREF
  char *v31; // [rsp+120h] [rbp-110h]
  char v32; // [rsp+130h] [rbp-100h] BYREF
  char *v33; // [rsp+160h] [rbp-D0h]
  char v34; // [rsp+170h] [rbp-C0h] BYREF

  v6 = *(_QWORD **)(*(_QWORD *)(a1 + 32) + 16LL);
  sub_3545000((__int64)v24, v6, a1, a4, a5, a6);
  v23 = sub_354E6C0((__int64)v24, (__int64)v6, v7, v8, v9, v10);
  if ( v33 != &v34 )
    _libc_free((unsigned __int64)v33);
  if ( v31 != &v32 )
    _libc_free((unsigned __int64)v31);
  v11 = v28;
  v12 = &v28[18 * v29];
  if ( v28 != v12 )
  {
    do
    {
      v12 -= 18;
      if ( (unsigned __int64 *)*v12 != v12 + 2 )
        _libc_free(*v12);
    }
    while ( v11 != v12 );
    v12 = v28;
  }
  if ( v12 != (unsigned __int64 *)&v30 )
    _libc_free((unsigned __int64)v12);
  v13 = v25;
  v14 = &v25[8 * v26];
  if ( v25 != v14 )
  {
    do
    {
      v15 = *((_QWORD *)v14 - 1);
      v14 -= 8;
      if ( v15 )
      {
        v16 = *(volatile signed __int32 **)(v15 + 32);
        if ( v16 )
        {
          if ( &_pthread_key_create )
          {
            v17 = _InterlockedExchangeAdd(v16 + 2, 0xFFFFFFFF);
          }
          else
          {
            v17 = *((_DWORD *)v16 + 2);
            *((_DWORD *)v16 + 2) = v17 - 1;
          }
          if ( v17 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 16LL))(v16);
            if ( &_pthread_key_create )
            {
              v22 = _InterlockedExchangeAdd(v16 + 3, 0xFFFFFFFF);
            }
            else
            {
              v22 = *((_DWORD *)v16 + 3);
              *((_DWORD *)v16 + 3) = v22 - 1;
            }
            if ( v22 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 24LL))(v16);
          }
        }
        v18 = *(volatile signed __int32 **)(v15 + 16);
        if ( v18 )
        {
          if ( &_pthread_key_create )
          {
            v19 = _InterlockedExchangeAdd(v18 + 2, 0xFFFFFFFF);
          }
          else
          {
            v19 = *((_DWORD *)v18 + 2);
            *((_DWORD *)v18 + 2) = v19 - 1;
          }
          if ( v19 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v18 + 16LL))(v18);
            if ( &_pthread_key_create )
            {
              v21 = _InterlockedExchangeAdd(v18 + 3, 0xFFFFFFFF);
            }
            else
            {
              v21 = *((_DWORD *)v18 + 3);
              *((_DWORD *)v18 + 3) = v21 - 1;
            }
            if ( v21 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v18 + 24LL))(v18);
          }
        }
        j_j___libc_free_0(v15);
      }
    }
    while ( v13 != v14 );
    v14 = v25;
  }
  if ( v14 != &v27 )
    _libc_free((unsigned __int64)v14);
  return v23;
}
