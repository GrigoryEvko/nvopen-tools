// Function: sub_6EA380
// Address: 0x6ea380
//
__int64 __fastcall sub_6EA380(__int64 a1, __int64 a2, int a3, int a4)
{
  int v5; // r13d
  __int64 result; // rax
  __int64 v9; // r12
  char v10; // al
  __int64 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rax
  char v14; // al
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  char v18; // al
  char v19; // [rsp+7h] [rbp-39h] BYREF
  __int64 *v20; // [rsp+8h] [rbp-38h] BYREF

  v5 = a2;
  if ( (*(_QWORD *)(a1 + 168) & 0x800700000LL) == 0x800100000LL )
  {
    if ( *(_BYTE *)(a1 + 177) )
      goto LABEL_3;
    v15 = *(_QWORD *)a1;
    v16 = *(_QWORD *)(v15 + 96);
    if ( (*(_BYTE *)(v16 + 81) & 8) != 0 && qword_4D03C50 && *(char *)(qword_4D03C50 + 18LL) < 0 )
    {
      sub_6E50A0();
    }
    else
    {
      *(_BYTE *)(v16 + 81) |= 8u;
      a2 = 1;
      sub_8AD0D0(v15, 1, 0);
    }
  }
  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 && !*(_BYTE *)(a1 + 177) )
  {
    if ( (unsigned int)sub_6EA1E0(a1) )
    {
      if ( dword_4F077BC )
      {
        v14 = *(_BYTE *)(a1 + 170);
        if ( (v14 & 0x10) != 0 && !**(_QWORD **)(a1 + 216) && (v14 & 0x20) == 0 )
          sub_5EB3F0((_QWORD *)a1);
      }
    }
  }
LABEL_3:
  if ( HIDWORD(qword_4F077B4) && (*(_BYTE *)(a1 + 174) & 8) != 0 )
  {
    sub_72F9F0(a1, 0, &v19, &v20);
    v9 = *v20;
    goto LABEL_15;
  }
  if ( dword_4F077C4 != 2 && !a4 || !(unsigned int)sub_6EA1E0(a1) )
    return 0;
  if ( (*(_BYTE *)(a1 + 89) & 4) != 0
    && (*(_QWORD *)(a1 + 168) & 0x400100000LL) == 0x100000
    && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 177LL) & 0x20) == 0 )
  {
    sub_8AC4A0(a1, a2);
  }
  v9 = sub_740200(a1);
  if ( !v9 )
  {
    if ( !dword_4F077BC
      || *(_BYTE *)(a1 + 177)
      || (*(_BYTE *)(a1 + 89) & 4) == 0
      || (*(_BYTE *)(a1 + 172) & 4) != 0
      || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 177LL) & 0x20) == 0
      || !(unsigned int)sub_8DBE70(*(_QWORD *)(a1 + 120)) )
    {
      return 0;
    }
    v9 = sub_724D50(12);
    v17 = sub_73E830(a1);
    sub_70FD90(v17, v9);
    if ( dword_4F077C4 != 2 )
    {
      if ( v9 )
      {
LABEL_13:
        if ( word_4D04898 )
        {
LABEL_14:
          if ( (*(_BYTE *)(a1 + 172) & 8) != 0 || (unsigned int)sub_8D32E0(*(_QWORD *)(a1 + 120)) )
            goto LABEL_15;
          goto LABEL_51;
        }
        goto LABEL_51;
      }
      return 0;
    }
    if ( word_4D04898 )
    {
      if ( v9 )
        goto LABEL_14;
      return 0;
    }
    if ( !dword_4D04964 )
    {
LABEL_60:
      if ( v9 )
        goto LABEL_51;
      return 0;
    }
LABEL_59:
    if ( (*(_BYTE *)(a1 + 176) & 2) == 0 )
      return 0;
    goto LABEL_60;
  }
  if ( dword_4F077C4 != 2 )
    goto LABEL_13;
  if ( word_4D04898 )
    goto LABEL_14;
  if ( dword_4D04964 )
    goto LABEL_59;
LABEL_51:
  v18 = *(_BYTE *)(v9 + 173);
  if ( v18 == 10
    || v18 == 6 && (!dword_4F077BC || *(_BYTE *)(v9 + 176) == 2 && *(_BYTE *)(*(_QWORD *)(v9 + 184) + 173LL) == 2) )
  {
    return 0;
  }
LABEL_15:
  if ( !v5 || !v9 )
    return v9;
  v10 = *(_BYTE *)(v9 + 173);
  v11 = *(_QWORD *)(v9 + 144);
  *(_QWORD *)(v9 + 144) = 0;
  if ( ((v10 - 10) & 0xFD) != 0 )
  {
    if ( v11 && a3 )
      result = sub_73A460(v9);
    else
      result = v9;
  }
  else
  {
    v12 = 32;
    if ( (*(_BYTE *)(v9 - 8) & 1) == 0 )
    {
      v13 = *(_QWORD *)(a1 + 48);
      if ( v13 )
      {
        if ( *(_DWORD *)(v13 + 164) != unk_4F07270 )
          v12 = 544;
      }
    }
    result = sub_740190(v9, 0, v12);
  }
  *(_QWORD *)(v9 + 144) = v11;
  return result;
}
