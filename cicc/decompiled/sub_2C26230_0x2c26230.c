// Function: sub_2C26230
// Address: 0x2c26230
//
void __fastcall sub_2C26230(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  unsigned int v7; // eax
  __int64 *v8; // rdx
  __int64 v9; // r12
  __int64 *v10; // rax
  char v11; // dl
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // r14
  __int64 v15; // rax
  const void *v16; // r8
  unsigned __int64 v17; // rdx
  const void *v18; // [rsp+8h] [rbp-D8h]
  __int64 *v19; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v20; // [rsp+18h] [rbp-C8h]
  _QWORD v21[6]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v23; // [rsp+58h] [rbp-88h]
  __int64 v24; // [rsp+60h] [rbp-80h]
  int v25; // [rsp+68h] [rbp-78h]
  unsigned __int8 v26; // [rsp+6Ch] [rbp-74h]
  char v27; // [rsp+70h] [rbp-70h] BYREF

  v6 = 1;
  v19 = v21;
  v22 = 0;
  v24 = 8;
  v25 = 0;
  v26 = 1;
  v21[0] = a1;
  v23 = (__int64 *)&v27;
  v20 = 0x600000001LL;
  v7 = 1;
  while ( v7 )
  {
    while ( 1 )
    {
      v8 = v19;
      v9 = v19[v7 - 1];
      LODWORD(v20) = v7 - 1;
      if ( (_BYTE)v6 )
      {
        v10 = v23;
        v8 = &v23[HIDWORD(v24)];
        if ( v23 != v8 )
        {
          while ( v9 != *v10 )
          {
            if ( v8 == ++v10 )
              goto LABEL_23;
          }
          goto LABEL_8;
        }
LABEL_23:
        if ( HIDWORD(v24) < (unsigned int)v24 )
          break;
      }
      sub_C8CC70((__int64)&v22, v9, (__int64)v8, v6, a5, a6);
      v6 = v26;
      if ( v11 )
        goto LABEL_15;
LABEL_8:
      v7 = v20;
      if ( !(_DWORD)v20 )
        goto LABEL_9;
    }
    ++HIDWORD(v24);
    *v8 = v9;
    ++v22;
LABEL_15:
    v12 = sub_2BF0490(v9);
    v13 = v12;
    if ( v12 && (unsigned __int8)sub_2C253E0(v12) )
    {
      v14 = *(unsigned int *)(v13 + 56);
      v15 = (unsigned int)v20;
      v16 = *(const void **)(v13 + 48);
      v17 = v14 + (unsigned int)v20;
      if ( v17 > HIDWORD(v20) )
      {
        v18 = *(const void **)(v13 + 48);
        sub_C8D5F0((__int64)&v19, v21, v17, 8u, (__int64)v16, a6);
        v15 = (unsigned int)v20;
        v16 = v18;
      }
      if ( 8 * v14 )
      {
        memcpy(&v19[v15], v16, 8 * v14);
        LODWORD(v15) = v20;
      }
      LODWORD(v20) = v15 + v14;
      sub_2C19E60((__int64 *)v13);
    }
    v7 = v20;
    v6 = v26;
  }
LABEL_9:
  if ( !(_BYTE)v6 )
    _libc_free((unsigned __int64)v23);
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
}
