// Function: sub_38599E0
// Address: 0x38599e0
//
__int64 __fastcall sub_38599E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r12d
  __int64 v5; // rax
  __int64 v6; // rbx
  int v7; // r14d
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // rcx
  char v11; // si
  unsigned __int64 v12; // r8
  __int64 v13; // rdx
  unsigned __int64 v14; // r13
  _QWORD *v15; // rdi
  _QWORD *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 result; // rax
  char v20; // r8
  int v23; // [rsp+30h] [rbp-2D0h]
  unsigned __int64 v24; // [rsp+30h] [rbp-2D0h]
  __int64 v25; // [rsp+38h] [rbp-2C8h] BYREF
  __int64 v26; // [rsp+90h] [rbp-270h] BYREF

  v25 = a1;
  if ( !a2 )
    return 0x7FFFFFFF;
  v23 = *(_DWORD *)(sub_1632FA0(*(_QWORD *)(a2 + 40)) + 4);
  v4 = sub_165AFC0(&v25);
  if ( v4 )
  {
    v5 = v25;
    v6 = 0;
    v7 = 0;
    while ( 1 )
    {
      v14 = v5 & 0xFFFFFFFFFFFFFFF8LL;
      v15 = (_QWORD *)((v5 & 0xFFFFFFFFFFFFFFF8LL) + 56);
      if ( (v5 & 4) != 0 )
      {
        if ( (unsigned __int8)sub_1560290(v15, v7, 6) )
          goto LABEL_20;
        v8 = *(_QWORD *)(v14 - 24);
        if ( !*(_BYTE *)(v8 + 16) )
          goto LABEL_6;
LABEL_11:
        v5 = v25;
        ++v7;
        v6 += 24;
        v12 = v25 & 0xFFFFFFFFFFFFFFF8LL;
        v13 = (v25 >> 2) & 1;
        if ( v4 == v7 )
          goto LABEL_12;
      }
      else
      {
        if ( !(unsigned __int8)sub_1560290(v15, v7, 6) )
        {
          v8 = *(_QWORD *)(v14 - 72);
          if ( *(_BYTE *)(v8 + 16) )
            goto LABEL_11;
LABEL_6:
          v26 = *(_QWORD *)(v8 + 112);
          v9 = sub_1560290(&v26, v7, 6);
          v10 = v25;
          v11 = v9;
          v5 = v25;
          v12 = v25 & 0xFFFFFFFFFFFFFFF8LL;
          v13 = (v25 >> 2) & 1;
          if ( !v11 )
            goto LABEL_7;
          goto LABEL_21;
        }
LABEL_20:
        v10 = v25;
LABEL_21:
        v5 = v10;
        v12 = v10 & 0xFFFFFFFFFFFFFFF8LL;
        v13 = (v10 >> 2) & 1;
        if ( *(_DWORD *)(**(_QWORD **)((v10 & 0xFFFFFFFFFFFFFFF8LL)
                                     + v6
                                     - 24LL * (*(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))
                       + 8LL) >> 8 != v23 )
          return 0x7FFFFFFF;
LABEL_7:
        ++v7;
        v6 += 24;
        if ( v4 == v7 )
          goto LABEL_12;
      }
    }
  }
  v12 = v25 & 0xFFFFFFFFFFFFFFF8LL;
  v13 = (v25 >> 2) & 1;
LABEL_12:
  v24 = v12;
  v16 = (_QWORD *)(v12 + 56);
  if ( (_BYTE)v13 )
  {
    if ( (unsigned __int8)sub_1560260(v16, -1, 3) )
      goto LABEL_29;
    v17 = *(_QWORD *)(v24 - 24);
    if ( *(_BYTE *)(v17 + 16) )
    {
LABEL_16:
      v18 = *(_QWORD *)(*(_QWORD *)((v25 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 56LL);
      if ( sub_14A3770(a4, v18, a2)
        && (unsigned __int8)sub_1560F60(v18, a2)
        && !(unsigned __int8)sub_1560180(v18 + 112, 35)
        && ((unsigned __int8)sub_15E4640(v18) || !(unsigned __int8)sub_15E4640(a2)) )
      {
        __asm { jmp     rax }
      }
      return 0x7FFFFFFF;
    }
  }
  else
  {
    if ( (unsigned __int8)sub_1560260(v16, -1, 3) )
      goto LABEL_29;
    v17 = *(_QWORD *)(v24 - 72);
    if ( *(_BYTE *)(v17 + 16) )
      goto LABEL_16;
  }
  v26 = *(_QWORD *)(v17 + 112);
  if ( !(unsigned __int8)sub_1560260(&v26, -1, 3) )
    goto LABEL_16;
LABEL_29:
  v20 = sub_3850BA0(a2);
  result = 0x80000000LL;
  if ( !v20 )
    return 0x7FFFFFFF;
  return result;
}
