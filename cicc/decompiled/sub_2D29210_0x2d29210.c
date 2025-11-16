// Function: sub_2D29210
// Address: 0x2d29210
//
__int64 __fastcall sub_2D29210(__int64 a1, __int64 a2, __int64 *a3)
{
  int v4; // r12d
  unsigned int v5; // r8d
  bool v7; // zf
  __int64 v8; // r14
  int v9; // r12d
  __int64 v10; // r15
  unsigned int v11; // eax
  bool v12; // al
  __int64 v13; // [rsp+20h] [rbp-C0h]
  int v14; // [rsp+28h] [rbp-B8h]
  unsigned int v15; // [rsp+2Ch] [rbp-B4h]
  int v16; // [rsp+3Ch] [rbp-A4h] BYREF
  __int64 v17; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v18; // [rsp+48h] [rbp-98h] BYREF
  _QWORD v19[3]; // [rsp+50h] [rbp-90h] BYREF
  char v20; // [rsp+68h] [rbp-78h]
  __int64 v21; // [rsp+70h] [rbp-70h]
  _QWORD v22[12]; // [rsp+80h] [rbp-60h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v7 = *(_BYTE *)(a2 + 24) == 0;
    v20 = 0;
    v19[0] = 0;
    v8 = *(_QWORD *)(a1 + 8);
    v21 = 0;
    memset(v22, 0, 24);
    v22[3] = 1;
    v22[4] = 0;
    v16 = 0;
    if ( !v7 )
      v16 = *(unsigned __int16 *)(a2 + 16) | (*(_DWORD *)(a2 + 8) << 16);
    v9 = v4 - 1;
    v18 = *(_QWORD *)(a2 + 32);
    v17 = *(_QWORD *)a2;
    v14 = 1;
    v13 = 0;
    v15 = v9 & sub_F11290(&v17, &v16, &v18);
    while ( 1 )
    {
      v10 = v8 + 56LL * v15;
      LOBYTE(v11) = sub_F34140(a2, v10);
      v5 = v11;
      if ( (_BYTE)v11 )
      {
        *a3 = v10;
        return v5;
      }
      if ( sub_F34140(v10, (__int64)v19) )
        break;
      v12 = sub_F34140(v10, (__int64)v22);
      if ( !v13 )
      {
        if ( !v12 )
          v10 = 0;
        v13 = v10;
      }
      v15 = v9 & (v14 + v15);
      ++v14;
    }
    v5 = 0;
    if ( v13 )
      v10 = v13;
    *a3 = v10;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
  return v5;
}
