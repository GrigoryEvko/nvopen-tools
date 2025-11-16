// Function: sub_2151550
// Address: 0x2151550
//
__int64 __fastcall sub_2151550(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v8; // rdx
  const char *v9; // rax
  size_t v10; // rdx
  _BYTE *v11; // rdi
  char *v12; // rsi
  _BYTE *v13; // rax
  size_t v14; // r14
  __int64 v15; // r15
  __int64 v16; // rax
  _QWORD *v17; // rbx
  char v18; // r13
  __int64 v19; // rax
  __int64 v20; // r15
  const char *v21; // rax
  size_t v22; // rdx
  __int64 v23; // rax
  __int64 v24; // [rsp-40h] [rbp-40h] BYREF

  result = *(_QWORD *)(a2 - 24);
  if ( !result )
    return result;
  if ( *(_BYTE *)(result + 16) <= 3u )
  {
    if ( *(_DWORD *)(a1[29] + 952LL) == 1 )
      sub_214CAD0((_BYTE *)a3, a4);
    v8 = *(_QWORD *)(a4 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v8) <= 5 )
    {
      sub_16E7EE0(a4, ".func ", 6u);
      if ( !(unsigned __int8)sub_1C2FA50(a3) )
        goto LABEL_9;
    }
    else
    {
      *(_DWORD *)v8 = 1853187630;
      *(_WORD *)(v8 + 4) = 8291;
      *(_QWORD *)(a4 + 24) += 6LL;
      if ( !(unsigned __int8)sub_1C2FA50(a3) )
      {
LABEL_9:
        sub_214D1D0((__int64)a1, **(_QWORD **)(*(_QWORD *)(a3 + 24) + 16LL), a3, a4);
        v9 = sub_1649960(a2);
        v11 = *(_BYTE **)(a4 + 24);
        v12 = (char *)v9;
        v13 = *(_BYTE **)(a4 + 16);
        v14 = v10;
        if ( v13 - v11 < v10 )
        {
          v15 = sub_16E7EE0(a4, v12, v10);
          v13 = *(_BYTE **)(v15 + 16);
          v11 = *(_BYTE **)(v15 + 24);
        }
        else
        {
          v15 = a4;
          if ( v10 )
          {
            memcpy(v11, v12, v10);
            v13 = *(_BYTE **)(a4 + 16);
            v11 = (_BYTE *)(v14 + *(_QWORD *)(a4 + 24));
            *(_QWORD *)(a4 + 24) = v11;
          }
        }
        if ( v11 == v13 )
        {
          sub_16E7EE0(v15, "\n", 1u);
        }
        else
        {
          *v11 = 10;
          ++*(_QWORD *)(v15 + 24);
        }
        goto LABEL_14;
      }
    }
    sub_214C940(a3, a4);
    goto LABEL_9;
  }
  if ( *(_WORD *)(result + 18) != 47 )
    return result;
  v19 = *(_QWORD *)result;
  if ( *(_BYTE *)(v19 + 8) != 15 )
    BUG();
  v20 = **(_QWORD **)(v19 + 16);
  if ( *(_BYTE *)(v20 + 8) != 12 )
    v20 = 0;
  if ( *(_DWORD *)(a1[29] + 952LL) == 1 )
    sub_214CAD0((_BYTE *)a3, a4);
  sub_1263B40(a4, ".func ");
  if ( (unsigned __int8)sub_1C2FA50(a3) )
    sub_214C940(a3, a4);
  sub_214D1D0((__int64)a1, **(_QWORD **)(v20 + 16), a3, a4);
  v21 = sub_1649960(a2);
  v23 = sub_1549FF0(a4, v21, v22);
  sub_1263B40(v23, "\n");
LABEL_14:
  sub_21502D0(a1, a3, a4);
  v16 = a3;
  v17 = (_QWORD *)(a3 & 0xFFFFFFFFFFFFFFF8LL);
  v18 = (v16 >> 2) & 1;
  if ( v18 )
  {
    if ( (unsigned __int8)sub_1560260(v17 + 7, -1, 29) )
      goto LABEL_40;
  }
  else
  {
    if ( v17 )
    {
      result = sub_1560180((__int64)(v17 + 14), 29);
      if ( !(_BYTE)result )
        return result;
LABEL_35:
      result = **(_QWORD **)(v17[3] + 16LL);
      if ( *(_BYTE *)(result + 8) )
        return result;
      goto LABEL_36;
    }
    if ( (unsigned __int8)sub_1560260((_QWORD *)0x38, -1, 29) )
    {
LABEL_28:
      result = **(_QWORD **)(MEMORY[0x40] + 16LL);
      if ( *(_BYTE *)(result + 8) )
        return result;
LABEL_36:
      result = sub_1C2F070((__int64)v17);
      if ( !(_BYTE)result )
        return sub_1263B40(a4, ".noreturn ");
      return result;
    }
  }
  result = *(v17 - 3);
  if ( *(_BYTE *)(result + 16) )
    return result;
  v24 = *(_QWORD *)(result + 112);
  result = sub_1560260(&v24, -1, 29);
  if ( !(_BYTE)result )
    return result;
  if ( !v18 )
  {
    if ( v17 )
      goto LABEL_35;
    goto LABEL_28;
  }
LABEL_40:
  result = **(_QWORD **)(v17[8] + 16LL);
  if ( !*(_BYTE *)(result + 8) )
    return sub_1263B40(a4, ".noreturn ");
  return result;
}
