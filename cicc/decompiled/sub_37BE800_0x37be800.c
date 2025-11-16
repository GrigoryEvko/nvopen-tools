// Function: sub_37BE800
// Address: 0x37be800
//
__int64 __fastcall sub_37BE800(__int64 a1, __int64 a2, __int64 *a3)
{
  int v4; // r12d
  __int64 v6; // rax
  bool v7; // zf
  int v8; // r12d
  int v9; // eax
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // r15
  char v13; // al
  bool v14; // al
  __int64 v15; // [rsp+10h] [rbp-D0h]
  __int64 v16; // [rsp+18h] [rbp-C8h]
  __int64 v17; // [rsp+20h] [rbp-C0h]
  int v18; // [rsp+28h] [rbp-B8h]
  unsigned int v19; // [rsp+2Ch] [rbp-B4h]
  int v20; // [rsp+3Ch] [rbp-A4h] BYREF
  __int64 v21; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v22; // [rsp+48h] [rbp-98h] BYREF
  _QWORD v23[3]; // [rsp+50h] [rbp-90h] BYREF
  char v24; // [rsp+68h] [rbp-78h]
  __int64 v25; // [rsp+70h] [rbp-70h]
  _QWORD v26[12]; // [rsp+80h] [rbp-60h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = *(_BYTE *)(a2 + 24) == 0;
    v24 = 0;
    v23[0] = 0;
    v17 = v6;
    v25 = 0;
    memset(v26, 0, 24);
    v26[3] = 1;
    v26[4] = 0;
    v20 = 0;
    if ( !v7 )
      v20 = *(unsigned __int16 *)(a2 + 16) | (*(_DWORD *)(a2 + 8) << 16);
    v8 = v4 - 1;
    v22 = *(_QWORD *)(a2 + 32);
    v21 = *(_QWORD *)a2;
    v9 = sub_F11290(&v21, &v20, &v22);
    v10 = *(_QWORD *)a2;
    v18 = 1;
    v15 = 0;
    v19 = v8 & v9;
    while ( 1 )
    {
      v11 = 48LL * v19;
      v12 = v17 + v11;
      if ( v10 == *(_QWORD *)(v17 + v11) )
      {
        v13 = *(_BYTE *)(a2 + 24);
        if ( v13 == *(_BYTE *)(v12 + 24)
          && (!v13 || *(_QWORD *)(a2 + 8) == *(_QWORD *)(v12 + 8) && *(_QWORD *)(a2 + 16) == *(_QWORD *)(v12 + 16))
          && *(_QWORD *)(a2 + 32) == *(_QWORD *)(v12 + 32) )
        {
          *a3 = v12;
          return 1;
        }
      }
      v16 = v10;
      if ( sub_F34140(v17 + v11, (__int64)v23) )
        break;
      v14 = sub_F34140(v12, (__int64)v26);
      v10 = v16;
      if ( !v15 )
      {
        if ( !v14 )
          v12 = 0;
        v15 = v12;
      }
      v19 = v8 & (v18 + v19);
      ++v18;
    }
    if ( v15 )
      v12 = v15;
    *a3 = v12;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
