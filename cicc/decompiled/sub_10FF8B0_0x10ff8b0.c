// Function: sub_10FF8B0
// Address: 0x10ff8b0
//
_QWORD *__fastcall sub_10FF8B0(char *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4; // rdx
  _QWORD *v5; // r9
  __int64 v7; // r15
  __int64 v8; // r12
  _BYTE *v9; // rsi
  __int64 v10; // r14
  _BYTE *v11; // rdi
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r14
  _QWORD *v15; // rax
  char v16; // [rsp-119h] [rbp-119h]
  __int64 v17; // [rsp-110h] [rbp-110h]
  unsigned __int8 v18; // [rsp-108h] [rbp-108h]
  _QWORD *v19; // [rsp-108h] [rbp-108h]
  _QWORD v20[2]; // [rsp-F8h] [rbp-F8h] BYREF
  _BYTE *v21; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v22; // [rsp-E0h] [rbp-E0h]
  _BYTE v23[64]; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 v24; // [rsp-98h] [rbp-98h] BYREF
  __int16 *v25; // [rsp-90h] [rbp-90h]
  __int64 v26; // [rsp-88h] [rbp-88h]
  int v27; // [rsp-80h] [rbp-80h]
  char v28; // [rsp-7Ch] [rbp-7Ch]
  __int16 v29; // [rsp-78h] [rbp-78h] BYREF

  v2 = *((_QWORD *)a1 - 4);
  if ( *(_BYTE *)v2 != 91 )
    return 0;
  v4 = *(_QWORD *)(v2 + 16);
  if ( !v4 )
    return 0;
  v5 = *(_QWORD **)(v4 + 8);
  if ( v5 )
    return 0;
  v7 = *((_QWORD *)a1 + 1);
  v8 = v7;
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
    v8 = **(_QWORD **)(v7 + 16);
  v9 = *(_BYTE **)(v2 - 96);
  v10 = *(_QWORD *)(v2 - 64);
  v17 = *(_QWORD *)(v2 - 32);
  v18 = *a1;
  if ( (unsigned __int8)(*v9 - 12) <= 1u )
    goto LABEL_19;
  if ( (unsigned __int8)(*v9 - 9) <= 2u )
  {
    v28 = 1;
    v25 = &v29;
    v20[1] = &v21;
    v21 = v23;
    v24 = 0;
    v26 = 8;
    v27 = 0;
    v22 = 0x800000000LL;
    v20[0] = &v24;
    v16 = sub_AA8FD0(v20, (__int64)v9);
    if ( v16 )
    {
      while ( 1 )
      {
        v11 = v21;
        if ( !(_DWORD)v22 )
          break;
        v9 = *(_BYTE **)&v21[8 * (unsigned int)v22 - 8];
        LODWORD(v22) = v22 - 1;
        if ( !(unsigned __int8)sub_AA8FD0(v20, (__int64)v9) )
          goto LABEL_21;
      }
    }
    else
    {
LABEL_21:
      v16 = 0;
      v11 = v21;
    }
    if ( v11 != v23 )
      _libc_free(v11, v9);
    if ( !v28 )
      _libc_free(v25, v9);
    v5 = 0;
    if ( v16 )
    {
LABEL_19:
      v12 = sub_ACA8A0((__int64 **)v7);
      v29 = 257;
      v13 = sub_10FF770(a2, (unsigned int)v18 - 29, v10, v8, (__int64)&v24, 0, (int)v21, 0);
      v29 = 257;
      v14 = v13;
      v15 = sub_BD2C40(72, 3u);
      v5 = v15;
      if ( v15 )
      {
        v19 = v15;
        sub_B4DFA0((__int64)v15, v12, v14, v17, (__int64)&v24, (__int64)v15, 0, 0);
        return v19;
      }
    }
  }
  return v5;
}
