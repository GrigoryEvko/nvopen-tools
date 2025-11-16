// Function: sub_2914D90
// Address: 0x2914d90
//
__int64 __fastcall sub_2914D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r15
  __int64 *v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r15
  __int64 v17; // r14
  __int64 v18; // r15
  __int64 *v19; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v25; // [rsp+18h] [rbp-98h]
  _BYTE v26[32]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v27; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v28; // [rsp+48h] [rbp-68h]
  __int64 v29; // [rsp+50h] [rbp-60h]
  int v30; // [rsp+58h] [rbp-58h]
  char v31; // [rsp+5Ch] [rbp-54h]
  char v32; // [rsp+60h] [rbp-50h] BYREF

  *(_QWORD *)(a1 + 8) = 0x600000000LL;
  *(_QWORD *)a1 = a1 + 16;
  v6 = *(_QWORD *)(a2 + 16);
  v28 = (__int64 *)&v32;
  v25 = 0x400000000LL;
  v27 = 0;
  v29 = 4;
  v30 = 0;
  v31 = 1;
  v24 = (__int64 *)v26;
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 24);
LABEL_3:
    v8 = v28;
    v9 = HIDWORD(v29);
    v10 = &v28[HIDWORD(v29)];
    if ( v28 == v10 )
    {
LABEL_29:
      if ( HIDWORD(v29) >= (unsigned int)v29 )
        goto LABEL_9;
      ++HIDWORD(v29);
      *v10 = v7;
      ++v27;
LABEL_10:
      v11 = (unsigned int)v25;
      v9 = HIDWORD(v25);
      v12 = (unsigned int)v25 + 1LL;
      if ( v12 > HIDWORD(v25) )
      {
        sub_C8D5F0((__int64)&v24, v26, v12, 8u, a5, a6);
        v11 = (unsigned int)v25;
      }
      v10 = v24;
      v24[v11] = v7;
      LODWORD(v25) = v25 + 1;
      v6 = *(_QWORD *)(v6 + 8);
      if ( v6 )
        goto LABEL_8;
    }
    else
    {
      while ( v7 != *v8 )
      {
        if ( v10 == ++v8 )
          goto LABEL_29;
      }
      while ( 1 )
      {
        v6 = *(_QWORD *)(v6 + 8);
        if ( !v6 )
          break;
LABEL_8:
        v7 = *(_QWORD *)(v6 + 24);
        if ( v31 )
          goto LABEL_3;
LABEL_9:
        sub_C8CC70((__int64)&v27, v7, (__int64)v10, v9, a5, a6);
        if ( (_BYTE)v10 )
          goto LABEL_10;
      }
    }
    v13 = v25;
    while ( (_DWORD)v25 )
    {
      while ( 1 )
      {
        v14 = v24;
        v15 = v13;
        v16 = v24[v13 - 1];
        LODWORD(v25) = v13 - 1;
        if ( *(_BYTE *)v16 == 63 )
          break;
        if ( (unsigned __int8)(*(_BYTE *)v16 - 78) <= 1u )
          goto LABEL_16;
LABEL_39:
        v23 = *(unsigned int *)(a1 + 8);
        if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, (const void *)(a1 + 16), v23 + 1, 8u, a5, a6);
          v23 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v23) = v16;
        v13 = v25;
        ++*(_DWORD *)(a1 + 8);
        if ( !v13 )
          goto LABEL_24;
      }
      if ( !(unsigned __int8)sub_B4DCF0(v16) )
        goto LABEL_39;
LABEL_16:
      v17 = *(_QWORD *)(v16 + 16);
      if ( v17 )
      {
        while ( 1 )
        {
          v18 = *(_QWORD *)(v17 + 24);
          if ( v31 )
          {
            v19 = v28;
            v15 = HIDWORD(v29);
            v14 = &v28[HIDWORD(v29)];
            if ( v28 != v14 )
            {
              while ( v18 != *v19 )
              {
                if ( v14 == ++v19 )
                  goto LABEL_36;
              }
              goto LABEL_22;
            }
LABEL_36:
            if ( HIDWORD(v29) < (unsigned int)v29 )
            {
              ++HIDWORD(v29);
              *v14 = v18;
              ++v27;
              goto LABEL_32;
            }
          }
          sub_C8CC70((__int64)&v27, *(_QWORD *)(v17 + 24), (__int64)v14, v15, a5, a6);
          if ( (_BYTE)v14 )
          {
LABEL_32:
            v21 = (unsigned int)v25;
            v15 = HIDWORD(v25);
            v22 = (unsigned int)v25 + 1LL;
            if ( v22 > HIDWORD(v25) )
            {
              sub_C8D5F0((__int64)&v24, v26, v22, 8u, a5, a6);
              v21 = (unsigned int)v25;
            }
            v14 = v24;
            v24[v21] = v18;
            LODWORD(v25) = v25 + 1;
            v17 = *(_QWORD *)(v17 + 8);
            if ( !v17 )
              break;
          }
          else
          {
LABEL_22:
            v17 = *(_QWORD *)(v17 + 8);
            if ( !v17 )
              break;
          }
        }
      }
      v13 = v25;
    }
LABEL_24:
    if ( v24 != (__int64 *)v26 )
      _libc_free((unsigned __int64)v24);
  }
  if ( !v31 )
    _libc_free((unsigned __int64)v28);
  return a1;
}
