// Function: sub_A751C0
// Address: 0xa751c0
//
__int64 __fastcall sub_A751C0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  char v4; // r14
  bool v6; // zf
  int v7; // eax
  int v8; // edx
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // [rsp+8h] [rbp-48h] BYREF
  __int64 v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = a4 & 1;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = a1 + 24;
  *(_QWORD *)(a1 + 48) = a1 + 24;
  *(_QWORD *)(a1 + 56) = 0;
  v6 = *(_BYTE *)(a2 + 8) == 12;
  v12 = a3;
  if ( v6 )
  {
LABEL_27:
    v13[0] = sub_A734C0(&v12, 97);
    if ( v13[0] )
    {
      v11 = sub_A72AA0(v13);
      if ( (unsigned int)sub_BCB060(a2) != *(_DWORD *)(v11 + 8) )
        *(_QWORD *)(a1 + 8) |= 0x200000000uLL;
    }
    if ( *(_BYTE *)(a2 + 8) == 14 )
    {
LABEL_33:
      if ( !v4 )
        return a1;
      goto LABEL_16;
    }
    if ( !v4 )
    {
LABEL_9:
      if ( (a4 & 2) == 0 )
        goto LABEL_10;
      goto LABEL_20;
    }
    v10 = *(_QWORD *)(a1 + 8);
LABEL_25:
    *(_QWORD *)a1 |= 0xC080000400200uLL;
    *(_QWORD *)(a1 + 8) = v10 | 0x40E002000LL;
    if ( (a4 & 2) == 0 )
    {
LABEL_10:
      v8 = *(unsigned __int8 *)(a2 + 8);
      if ( (unsigned int)(v8 - 17) <= 1 )
        LOBYTE(v8) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
      if ( (_BYTE)v8 != 14 )
      {
        if ( !v4 )
          return a1;
        *(_QWORD *)(a1 + 8) |= 0x400000uLL;
        goto LABEL_16;
      }
      goto LABEL_33;
    }
LABEL_20:
    *(_QWORD *)(a1 + 8) |= 0x3F0400uLL;
    *(_QWORD *)a1 |= 0x200004uLL;
    goto LABEL_10;
  }
  if ( !v4 )
  {
    if ( (a4 & 2) == 0 )
      goto LABEL_4;
    goto LABEL_21;
  }
  *(_QWORD *)a1 = 2;
  if ( (a4 & 2) != 0 )
  {
LABEL_21:
    *(_QWORD *)(a1 + 8) = 0x8000;
    *(_QWORD *)a1 |= 0x40000000000000uLL;
  }
LABEL_4:
  v7 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v7 - 17) <= 1 )
    LOBYTE(v7) = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
  if ( (_BYTE)v7 == 12 )
    goto LABEL_27;
  if ( !v4 )
  {
    if ( *(_BYTE *)(a2 + 8) == 14 )
      return a1;
    goto LABEL_9;
  }
  v10 = *(_QWORD *)(a1 + 8) | 0x200000000LL;
  *(_QWORD *)(a1 + 8) = v10;
  if ( *(_BYTE *)(a2 + 8) != 14 )
    goto LABEL_25;
LABEL_16:
  if ( !(unsigned __int8)sub_A750C0(a2) )
    *(_QWORD *)(a1 + 8) |= 0x20000000uLL;
  if ( *(_BYTE *)(a2 + 8) == 7 )
    *(_QWORD *)a1 |= 0x10000000000uLL;
  return a1;
}
