// Function: sub_FD7E80
// Address: 0xfd7e80
//
char __fastcall sub_FD7E80(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7[3]; // [rsp+8h] [rbp-18h] BYREF

  v7[0] = a2;
  v2 = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 40) == v2 )
  {
    *(_DWORD *)(a1 + 64) = (*(_DWORD *)(a1 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(a1 + 64) & 0xF8000000;
    if ( v2 != *(_QWORD *)(a1 + 56) )
      goto LABEL_3;
LABEL_15:
    sub_FD7BE0((unsigned __int64 **)(a1 + 40), (char *)v2, v7);
    v3 = v7[0];
    goto LABEL_9;
  }
  if ( v2 == *(_QWORD *)(a1 + 56) )
    goto LABEL_15;
LABEL_3:
  v3 = v7[0];
  if ( v2 )
  {
    *(_QWORD *)v2 = 0;
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = v3;
    if ( v3 != 0 && v3 != -4096 && v3 != -8192 )
      sub_BD73F0(v2);
    v2 = *(_QWORD *)(a1 + 48);
    v3 = v7[0];
  }
  *(_QWORD *)(a1 + 48) = v2 + 24;
LABEL_9:
  LOBYTE(v4) = sub_B46490(v3);
  if ( !(_BYTE)v4
    || (LOBYTE(v4) = sub_D222C0(v3), (_BYTE)v4)
    || (LOBYTE(v4) = v7[0], !*(_QWORD *)(v7[0] + 16))
    && *(_BYTE *)v7[0] == 85
    && (v5 = *(_QWORD *)(v7[0] - 32)) != 0
    && !*(_BYTE *)v5
    && (v4 = *(_QWORD *)(v7[0] + 80), *(_QWORD *)(v5 + 24) == v4)
    && *(_DWORD *)(v5 + 36) == 205 )
  {
    *(_BYTE *)(a1 + 67) |= 0x40u;
    *(_BYTE *)(a1 + 67) |= 0x10u;
  }
  else
  {
    *(_BYTE *)(a1 + 67) |= 0x70u;
  }
  return v4;
}
