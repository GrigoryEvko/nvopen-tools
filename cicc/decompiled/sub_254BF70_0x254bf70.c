// Function: sub_254BF70
// Address: 0x254bf70
//
__int64 __fastcall sub_254BF70(__int64 a1, __int64 a2)
{
  int v3; // edx
  char v4; // r15
  unsigned __int64 v5; // r14
  char v6; // dl
  __int64 v7; // rdx
  int v9; // edx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // [rsp+0h] [rbp-60h]
  char v18; // [rsp+Fh] [rbp-51h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  char v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  __int64 v23; // [rsp+28h] [rbp-38h]

  while ( 1 )
  {
    v3 = *(unsigned __int8 *)(a1 + 8);
    if ( (_BYTE)v3 != 12
      && (unsigned __int8)v3 > 3u
      && (_BYTE)v3 != 5
      && (v3 & 0xFD) != 4
      && (v3 & 0xFB) != 0xA
      && ((unsigned __int8)(*(_BYTE *)(a1 + 8) - 15) > 3u && v3 != 20 || !(unsigned __int8)sub_BCEBA0(a1, 0)) )
    {
      return 0;
    }
    v4 = sub_AE5020(a2, a1);
    v5 = (unsigned __int64)(sub_9208B0(a2, a1) + 7) >> 3;
    v20 = v6;
    v22 = sub_9208B0(a2, a1);
    v23 = v7;
    if ( v22 != 8 * (((1LL << v4) + v5 - 1) >> v4 << v4) || (_BYTE)v23 != v20 )
      return 0;
    v9 = *(unsigned __int8 *)(a1 + 8);
    if ( (unsigned int)(v9 - 17) > 1 && (_BYTE)v9 != 16 )
      break;
    a1 = *(_QWORD *)(a1 + 24);
  }
  if ( (_BYTE)v9 == 15 )
  {
    v10 = sub_AE4AC0(a2, a1);
    v11 = *(unsigned int *)(a1 + 12);
    if ( (_DWORD)v11 )
    {
      v21 = 0;
      v12 = 0;
      v17 = 8 * v11;
      while ( 1 )
      {
        v19 = v12;
        v13 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + v12);
        if ( !(unsigned __int8)sub_254BF70(v13, a2) )
          break;
        v14 = 8LL * *(_QWORD *)(v10 + 2 * v19 + 24);
        LOBYTE(v23) = *(_BYTE *)(v10 + 2 * v19 + 32);
        v22 = v14;
        if ( sub_CA1930(&v22) != v21 )
          break;
        v18 = sub_AE5020(a2, v13);
        v15 = sub_9208B0(a2, v13);
        v23 = v16;
        v22 = 8 * (((1LL << v18) + ((unsigned __int64)(v15 + 7) >> 3) - 1) >> v18 << v18);
        v21 += sub_CA1930(&v22);
        v12 = v19 + 8;
        if ( v19 + 8 == v17 )
          return 1;
      }
      return 0;
    }
  }
  return 1;
}
