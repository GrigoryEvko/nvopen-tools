// Function: sub_1AD6F20
// Address: 0x1ad6f20
//
__int64 __fastcall sub_1AD6F20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rdx
  char v8; // al
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // rax
  unsigned int *v16; // rax
  __int64 v17; // rax
  __int64 v19; // [rsp+0h] [rbp-60h]
  _QWORD v21[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(a1 + 48);
  v19 = a1;
  if ( v3 == a1 + 40 )
    return 0;
  while ( 1 )
  {
    v4 = v3;
    v3 = *(_QWORD *)(v3 + 8);
    if ( *(_BYTE *)(v4 - 8) != 78 || (unsigned __int8)sub_1560260((_QWORD *)(v4 + 32), -1, 30) )
      goto LABEL_23;
    v5 = *(_QWORD *)(v4 - 48);
    v6 = *(_BYTE *)(v5 + 16);
    if ( !v6 )
      break;
    if ( v6 != 20 )
      goto LABEL_9;
LABEL_23:
    if ( a1 + 40 == v3 )
      return 0;
  }
  v21[0] = *(_QWORD *)(v5 + 112);
  if ( (unsigned __int8)sub_1560260(v21, -1, 30) )
    goto LABEL_23;
  v7 = *(_QWORD *)(v4 - 48);
  v8 = *(_BYTE *)(v7 + 16);
  if ( v8 == 20 || !v8 && (*(_DWORD *)(v7 + 36) & 0xFFFFFFFB) == 0x4B )
    goto LABEL_23;
LABEL_9:
  if ( *(char *)(v4 - 1) < 0 )
  {
    v9 = sub_1648A40(v4 - 24);
    v11 = v9 + v10;
    if ( *(char *)(v4 - 1) < 0 )
      v11 -= sub_1648A40(v4 - 24);
    v12 = v11 >> 4;
    if ( (_DWORD)v12 )
    {
      v13 = 0;
      v14 = 16LL * (unsigned int)v12;
      while ( 1 )
      {
        v15 = 0;
        if ( *(char *)(v4 - 1) < 0 )
          v15 = sub_1648A40(v4 - 24);
        v16 = (unsigned int *)(v13 + v15);
        if ( *(_DWORD *)(*(_QWORD *)v16 + 8LL) == 1 )
          break;
        v13 += 16;
        if ( v13 == v14 )
          goto LABEL_20;
      }
      v17 = sub_1AD6830(*(_QWORD *)(v4 - 24 + 24 * (v16[2] - (unsigned __int64)(*(_DWORD *)(v4 - 4) & 0xFFFFFFF))), a3);
      if ( v17 )
      {
        if ( *(_BYTE *)(v17 + 16) != 16 )
          goto LABEL_23;
      }
    }
  }
LABEL_20:
  sub_1AEFCD0(v4 - 24, a2);
  return v19;
}
