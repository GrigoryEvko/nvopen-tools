// Function: sub_34A1150
// Address: 0x34a1150
//
__int64 __fastcall sub_34A1150(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // r14
  int v4; // r13d
  bool v5; // zf
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // r15
  int v10; // eax
  char v11; // al
  bool v12; // al
  __int64 v13; // [rsp+10h] [rbp-D0h]
  __int64 v15; // [rsp+20h] [rbp-C0h]
  int v16; // [rsp+28h] [rbp-B8h]
  unsigned int v17; // [rsp+2Ch] [rbp-B4h]
  int v18; // [rsp+3Ch] [rbp-A4h] BYREF
  __int64 v19; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v20; // [rsp+48h] [rbp-98h] BYREF
  _QWORD v21[3]; // [rsp+50h] [rbp-90h] BYREF
  char v22; // [rsp+68h] [rbp-78h]
  __int64 v23; // [rsp+70h] [rbp-70h]
  _QWORD v24[12]; // [rsp+80h] [rbp-60h] BYREF

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v3 = a1 + 16;
    v4 = 7;
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 24);
    v3 = *(_QWORD *)(a1 + 16);
    v4 = v10 - 1;
    if ( !v10 )
    {
      *a3 = 0;
      return 0;
    }
  }
  v5 = *(_BYTE *)(a2 + 24) == 0;
  v22 = 0;
  v21[0] = 0;
  v23 = 0;
  memset(v24, 0, 24);
  v24[3] = 1;
  v24[4] = 0;
  v18 = 0;
  if ( !v5 )
    v18 = *(unsigned __int16 *)(a2 + 16) | (*(_DWORD *)(a2 + 8) << 16);
  v20 = *(_QWORD *)(a2 + 32);
  v19 = *(_QWORD *)a2;
  v6 = sub_F11290(&v19, &v18, &v20);
  v7 = *(_QWORD *)a2;
  v16 = 1;
  v13 = 0;
  v17 = v4 & v6;
  while ( 1 )
  {
    v8 = v3 + 72LL * v17;
    if ( *(_QWORD *)v8 == v7 )
    {
      v11 = *(_BYTE *)(a2 + 24);
      if ( v11 == *(_BYTE *)(v8 + 24)
        && (!v11 || *(_QWORD *)(a2 + 8) == *(_QWORD *)(v8 + 8) && *(_QWORD *)(a2 + 16) == *(_QWORD *)(v8 + 16))
        && *(_QWORD *)(a2 + 32) == *(_QWORD *)(v8 + 32) )
      {
        *a3 = v8;
        return 1;
      }
    }
    v15 = v7;
    if ( sub_F34140(v3 + 72LL * v17, (__int64)v21) )
      break;
    v12 = sub_F34140(v8, (__int64)v24);
    v7 = v15;
    if ( !v13 )
    {
      if ( !v12 )
        v8 = 0;
      v13 = v8;
    }
    v17 = v4 & (v16 + v17);
    ++v16;
  }
  if ( v13 )
    v8 = v13;
  *a3 = v8;
  return 0;
}
