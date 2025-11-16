// Function: sub_1E176D0
// Address: 0x1e176d0
//
__int64 __fastcall sub_1E176D0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int16 v4; // dx
  char v5; // al
  __int64 v6; // rax
  __int64 **v7; // r13
  __int64 v8; // r12
  __int64 *v9; // r15
  __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 **v18; // [rsp+8h] [rbp-68h]
  _QWORD v19[12]; // [rsp+10h] [rbp-60h] BYREF

  v3 = *(_QWORD *)(a1 + 16);
  if ( *(_WORD *)v3 == 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 8) != 0
    || ((v4 = *(_WORD *)(a1 + 46), (v4 & 4) != 0) || (v4 & 8) == 0
      ? (v5 = WORD1(*(_QWORD *)(v3 + 8)) & 1)
      : (v5 = sub_1E15D00(a1, 0x10000u, 1)),
        v5) )
  {
    v6 = *(unsigned __int8 *)(a1 + 49);
    if ( (_BYTE)v6 )
    {
      v7 = *(__int64 ***)(a1 + 56);
      v18 = &v7[v6];
      v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 56LL) + 56LL);
      while ( 1 )
      {
        v9 = *v7;
        if ( ((*v7)[4] & 6) != 0 )
          return 0;
        if ( ((*v7)[4] & 0x30) != 0x30 )
        {
          v10 = *v9;
          if ( (*v9 & 4) == 0 )
            goto LABEL_14;
          v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v11 )
            return 0;
          if ( !(*(unsigned __int8 (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v11 + 24LL))(v11, v8) )
            break;
        }
LABEL_20:
        if ( v18 == ++v7 )
          return 1;
      }
      v10 = *v9;
      if ( (*v9 & 4) != 0 )
        return 0;
LABEL_14:
      v12 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v12 )
        return 0;
      if ( !a2 )
        return 0;
      v13 = v9[5];
      v14 = v9[6];
      v19[0] = v12;
      v15 = v9[3];
      v16 = v9[7];
      v19[2] = v13;
      v19[3] = v14;
      v19[1] = v15;
      v19[4] = v16;
      if ( !(unsigned __int8)sub_134CBB0(a2, (__int64)v19, 0) )
        return 0;
      goto LABEL_20;
    }
  }
  return 0;
}
