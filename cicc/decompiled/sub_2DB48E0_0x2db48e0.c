// Function: sub_2DB48E0
// Address: 0x2db48e0
//
__int64 __fastcall sub_2DB48E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r15
  int v8; // r14d
  int v9; // eax
  int v11; // eax
  __int64 v12; // rax
  char v13[49]; // [rsp+Fh] [rbp-31h] BYREF

  v5 = sub_2E313E0(a2, a2, a3, a4, a5);
  v6 = *(_QWORD *)(a2 + 56);
  if ( v6 == v5 )
    return 1;
  v7 = v5;
  v8 = 0;
  while ( 1 )
  {
    v9 = *(unsigned __int16 *)(v6 + 68);
    if ( (unsigned __int16)(v9 - 14) > 4u )
    {
      if ( ++v8 > (unsigned int)qword_501D0E8 && !(_BYTE)qword_501D008 )
        break;
      if ( !*(_WORD *)(v6 + 68)
        || v9 == 68
        || (unsigned int)(v9 - 1) <= 1 && (*(_BYTE *)(*(_QWORD *)(v6 + 32) + 64LL) & 8) != 0 )
      {
        break;
      }
      v11 = *(_DWORD *)(v6 + 44);
      if ( (v11 & 4) != 0 || (v11 & 8) == 0 )
        v12 = (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) >> 19) & 1LL;
      else
        LOBYTE(v12) = sub_2E88A90(v6, 0x80000, 1);
      if ( (_BYTE)v12 )
        break;
      v13[0] = 1;
      if ( !(unsigned __int8)sub_2E8B400(v6, v13) || !(unsigned __int8)sub_2DB4530(a1, v6) )
        break;
    }
    if ( (*(_BYTE *)v6 & 4) != 0 )
    {
      v6 = *(_QWORD *)(v6 + 8);
      if ( v7 == v6 )
        return 1;
    }
    else
    {
      while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
        v6 = *(_QWORD *)(v6 + 8);
      v6 = *(_QWORD *)(v6 + 8);
      if ( v7 == v6 )
        return 1;
    }
  }
  return 0;
}
