// Function: sub_17D7AC0
// Address: 0x17d7ac0
//
unsigned __int64 __fastcall sub_17D7AC0(__int128 a1)
{
  __int64 v1; // r14
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 *v5; // rax
  unsigned int v6; // ebx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int v10; // eax
  unsigned __int64 result; // rax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // r14
  __int64 v15; // rsi
  __int64 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // [rsp+8h] [rbp-A8h]
  _BYTE v21[16]; // [rsp+10h] [rbp-A0h] BYREF
  __int16 v22; // [rsp+20h] [rbp-90h]
  __int64 v23[16]; // [rsp+30h] [rbp-80h] BYREF

  v2 = *((_QWORD *)&a1 + 1);
  v3 = *(_QWORD *)(*((_QWORD *)&a1 + 1) + 40LL);
  v4 = *(_QWORD *)(*((_QWORD *)&a1 + 1) + 32LL);
  if ( v4 == v3 + 40 || !v4 )
    *((_QWORD *)&a1 + 1) = 0;
  else
    *((_QWORD *)&a1 + 1) = v4 - 24;
  sub_17CE510((__int64)v23, *((__int64 *)&a1 + 1), 0, 0, 0);
  v5 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v2);
  v6 = 1 << (*(unsigned __int16 *)(v2 + 18) >> 1) >> 1;
  if ( *(_BYTE *)(a1 + 489) )
  {
    v7 = sub_17CFB40(a1, *(_QWORD *)(v2 - 24), v23, v5, v6);
    v1 = v8;
    v20 = sub_17D3810(v23, v7, "_msld");
    sub_15F8F50((__int64)v20, v6);
    *((_QWORD *)&a1 + 1) = v2;
    sub_17D4920(a1, (__int64 *)v2, (__int64)v20);
  }
  else
  {
    v15 = *(_QWORD *)v2;
    v16 = sub_17CD8D0((_QWORD *)a1, *(_QWORD *)v2);
    v18 = (__int64)v16;
    if ( v16 )
      v18 = sub_15A06D0((__int64 **)v16, v15, (__int64)v16, v17);
    *((_QWORD *)&a1 + 1) = v2;
    sub_17D4920(a1, (__int64 *)v2, v18);
  }
  if ( byte_4FA4600 )
  {
    *((_QWORD *)&a1 + 1) = *(_QWORD *)(v2 - 24);
    sub_17D5820(a1, v2);
  }
  if ( sub_15F32D0(v2) )
  {
    v10 = *(unsigned __int16 *)(v2 + 18);
    *((_QWORD *)&a1 + 1) = (v10 >> 7) & 7;
    switch ( (v10 >> 7) & 7 )
    {
      case 0u:
        v9 = 0;
        break;
      case 1u:
      case 2u:
      case 4u:
        v9 = 512;
        break;
      case 3u:
      case 7u:
        v9 = 896;
        break;
      case 5u:
      case 6u:
        v9 = 768;
        break;
    }
    *(_WORD *)(v2 + 18) = *(_WORD *)(v2 + 18) & 0x8000 | v9 | v10 & 0x7C7F;
  }
  result = *(_QWORD *)(a1 + 8);
  v12 = *(unsigned int *)(result + 156);
  if ( (_DWORD)v12 )
  {
    if ( *(_BYTE *)(a1 + 489) )
    {
      v22 = 257;
      v13 = sub_156E5B0(v23, v1, (__int64)v21);
      DWORD2(a1) = 4;
      if ( v6 >= 4 )
        DWORD2(a1) = v6;
      v14 = (__int64)v13;
      sub_15F8F50((__int64)v13, DWORD2(a1));
      result = sub_17D4B80(a1, v2, v14);
    }
    else
    {
      v19 = sub_15A06D0(*(__int64 ***)(result + 184), *((__int64 *)&a1 + 1), v12, v9);
      result = sub_17D4B80(a1, v2, v19);
    }
  }
  if ( v23[0] )
    return sub_161E7C0((__int64)v23, v23[0]);
  return result;
}
