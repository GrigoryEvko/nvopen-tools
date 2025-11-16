// Function: sub_D13D60
// Address: 0xd13d60
//
__int64 __fastcall sub_D13D60(__int64 a1, __int64 a2, int a3)
{
  unsigned int v3; // eax
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rsi
  __int64 result; // rax
  __int64 *v9; // r14
  unsigned int v10; // eax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int8 v15; // [rsp+Fh] [rbp-241h]
  unsigned int v16; // [rsp+10h] [rbp-240h]
  int v17; // [rsp+14h] [rbp-23Ch] BYREF
  __int64 v18; // [rsp+18h] [rbp-238h] BYREF
  __int64 v19; // [rsp+28h] [rbp-228h] BYREF
  _QWORD v20[4]; // [rsp+30h] [rbp-220h] BYREF
  _BYTE *v21; // [rsp+50h] [rbp-200h] BYREF
  __int64 v22; // [rsp+58h] [rbp-1F8h]
  _BYTE v23[160]; // [rsp+60h] [rbp-1F0h] BYREF
  __int64 v24; // [rsp+100h] [rbp-150h] BYREF
  _BYTE *v25; // [rsp+108h] [rbp-148h]
  __int64 v26; // [rsp+110h] [rbp-140h]
  int v27; // [rsp+118h] [rbp-138h]
  char v28; // [rsp+11Ch] [rbp-134h]
  _BYTE v29[304]; // [rsp+120h] [rbp-130h] BYREF

  v18 = a2;
  v17 = a3;
  if ( !a3 )
    v17 = qword_4F86B28;
  v21 = v23;
  v22 = 0x1400000000LL;
  v3 = sub_D138F0();
  if ( v3 > 0x14 )
    sub_C8D5F0((__int64)&v21, v23, v3, 8u, v5, v6);
  v7 = *(_QWORD *)(a1 + 16);
  v24 = 0;
  v20[0] = &v24;
  v20[1] = &v17;
  v25 = v29;
  v26 = 32;
  v27 = 0;
  v28 = 1;
  v20[2] = &v18;
  v20[3] = &v21;
  result = sub_D13750((__int64)v20, v7, (__int64)v29, v4, v5, v6);
  if ( (_BYTE)result )
  {
    v19 = v18;
    result = (unsigned int)v22;
    if ( (_DWORD)v22 )
    {
      while ( 1 )
      {
        v7 = a1;
        v9 = *(__int64 **)&v21[8 * (unsigned int)result - 8];
        LODWORD(v22) = result - 1;
        LOWORD(v10) = sub_D139D0(
                        v9,
                        a1,
                        (unsigned __int8 (__fastcall *)(__int64, unsigned __int8 *, __int64))sub_D138A0,
                        (__int64)&v19);
        v13 = v10;
        v14 = BYTE1(v10);
        if ( !(_BYTE)v10 )
          goto LABEL_8;
        v15 = BYTE1(v10);
        v7 = (__int64)v9;
        v16 = v10;
        result = (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)v18 + 32LL))(v18, v9);
        v13 = v16;
        v14 = v15;
        if ( (_DWORD)result != 1 )
          break;
        v13 = v15 & (unsigned __int8)~(_BYTE)v16 & 0xF;
        if ( (v15 & (unsigned __int8)~(_BYTE)v16 & 0xF) == 0 )
          goto LABEL_9;
LABEL_20:
        v7 = *(_QWORD *)(v9[3] + 16);
        result = sub_D13750((__int64)v20, v7, v13, v14, v11, v12);
        if ( !(_BYTE)result )
          goto LABEL_14;
LABEL_9:
        result = (unsigned int)v22;
        if ( !(_DWORD)v22 )
          goto LABEL_14;
      }
      if ( (_DWORD)result == 2 )
        goto LABEL_9;
      if ( !(_DWORD)result )
        goto LABEL_14;
LABEL_8:
      if ( !(_BYTE)v14 )
        goto LABEL_9;
      goto LABEL_20;
    }
  }
LABEL_14:
  if ( !v28 )
    result = _libc_free(v25, v7);
  if ( v21 != v23 )
    return _libc_free(v21, v7);
  return result;
}
