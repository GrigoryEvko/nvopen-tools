// Function: sub_165B7E0
// Address: 0x165b7e0
//
void __fastcall sub_165B7E0(__int64 *a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  _QWORD *v5; // rbx
  unsigned __int64 v6; // r14
  int i; // r12d
  unsigned __int64 v8; // r15
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r12
  _BYTE *v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rsi
  unsigned __int8 v15; // al
  __int64 v17; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v18[2]; // [rsp+20h] [rbp-50h] BYREF
  char v19; // [rsp+30h] [rbp-40h]
  char v20; // [rsp+31h] [rbp-3Fh]

  v4 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 20);
  v17 = a2;
  v5 = (_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (v4 & 0xFFFFFFF));
  v6 = sub_1389B50(&v17);
  if ( (_QWORD *)v6 == v5 )
    return;
  for ( i = 0; ; ++i )
  {
    if ( *v5 != a3 )
      goto LABEL_3;
    v8 = v17 & 0xFFFFFFFFFFFFFFF8LL;
    v9 = (_QWORD *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 56);
    if ( (v17 & 4) != 0 )
    {
      if ( (unsigned __int8)sub_1560290(v9, i, 54) )
        goto LABEL_3;
      v10 = *(_QWORD *)(v8 - 24);
      if ( *(_BYTE *)(v10 + 16) )
        break;
      goto LABEL_8;
    }
    if ( (unsigned __int8)sub_1560290(v9, i, 54) )
      goto LABEL_3;
    v10 = *(_QWORD *)(v8 - 72);
    if ( *(_BYTE *)(v10 + 16) )
      break;
LABEL_8:
    v18[0] = *(_QWORD *)(v10 + 112);
    if ( !(unsigned __int8)sub_1560290(v18, i, 54) )
      break;
LABEL_3:
    v5 += 3;
    if ( (_QWORD *)v6 == v5 )
      return;
  }
  v20 = 1;
  v18[0] = "swifterror value when used in a callsite should be marked with swifterror attribute";
  v19 = 3;
  v11 = *a1;
  if ( *a1 )
  {
    sub_16E2CE0(v18, *a1);
    v12 = *(_BYTE **)(v11 + 24);
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 16) )
    {
      sub_16E7DE0(v11, 10);
    }
    else
    {
      *(_QWORD *)(v11 + 24) = v12 + 1;
      *v12 = 10;
    }
    v13 = *a1;
    *((_BYTE *)a1 + 72) = 1;
    if ( v13 )
    {
      sub_164FA80(a1, a3);
      v14 = v17 & 0xFFFFFFFFFFFFFFF8LL;
      v15 = *(_BYTE *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 16);
      if ( v15 <= 0x17u )
      {
        v14 = 0;
      }
      else if ( v15 != 78 && v15 != 29 )
      {
        v14 = 0;
      }
      sub_164FA80(a1, v14);
    }
  }
  else
  {
    *((_BYTE *)a1 + 72) = 1;
  }
}
