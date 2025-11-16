// Function: sub_396F5A0
// Address: 0x396f5a0
//
__int64 __fastcall sub_396F5A0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  char v5; // r13
  __int64 v6; // rax
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  unsigned __int64 v11; // rsi
  _QWORD *v12; // rax
  _DWORD *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 (*v16)(); // rdx
  __int64 result; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  _DWORD *v20; // r8
  _DWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  _QWORD v24[2]; // [rsp+0h] [rbp-40h] BYREF
  char v25; // [rsp+10h] [rbp-30h]
  char v26; // [rsp+11h] [rbp-2Fh]

  *(_QWORD *)(a1 + 264) = a2;
  v3 = sub_396EAF0(a1, *a2);
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 304) = v3;
  *(_QWORD *)(a1 + 312) = v3;
  v4 = *(_QWORD *)(a1 + 240);
  *(_QWORD *)(a1 + 400) = 0;
  v5 = *(_BYTE *)(v4 + 73);
  v6 = sub_1626D20(*a2);
  if ( !v6 || *(_DWORD *)(*(_QWORD *)(v6 + 8 * (5LL - *(unsigned int *)(v6 + 8))) + 36LL) == 3 )
    goto LABEL_3;
  if ( sub_396B980((__int64)a2, *(_QWORD *)(a1 + 272)) )
  {
    v26 = 1;
    v24[0] = "func_begin";
    v25 = 3;
    v18 = sub_396F530(a1, (__int64)v24);
    *(_QWORD *)(a1 + 384) = v18;
    if ( !v5 )
      goto LABEL_3;
LABEL_20:
    *(_QWORD *)(a1 + 312) = v18;
    goto LABEL_3;
  }
  if ( v5 )
  {
    v26 = 1;
    v24[0] = "func_begin";
    v25 = 3;
    v18 = sub_396F530(a1, (__int64)v24);
    *(_QWORD *)(a1 + 384) = v18;
    goto LABEL_20;
  }
  if ( (*(_BYTE *)(a2[1] + 809) & 2) != 0 )
  {
    v26 = 1;
    v24[0] = "func_begin";
    v25 = 3;
    *(_QWORD *)(a1 + 384) = sub_396F530(a1, (__int64)v24);
  }
LABEL_3:
  v7 = *(__int64 **)(a1 + 8);
  v8 = *v7;
  v9 = v7[1];
  if ( v8 == v9 )
LABEL_34:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_4FC6AEC )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_34;
  }
  *(_QWORD *)(a1 + 296) = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(
                                        *(_QWORD *)(v8 + 8),
                                        &unk_4FC6AEC)
                                    + 232);
  v10 = a2[2];
  v11 = sub_16D5D50();
  v12 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_14;
  v13 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v14 = v12[2];
      v15 = v12[3];
      if ( v11 <= v12[4] )
        break;
      v12 = (_QWORD *)v12[3];
      if ( !v15 )
        goto LABEL_12;
    }
    v13 = v12;
    v12 = (_QWORD *)v12[2];
  }
  while ( v14 );
LABEL_12:
  if ( v13 == dword_4FA0208 )
    goto LABEL_14;
  if ( v11 < *((_QWORD *)v13 + 4) )
    goto LABEL_14;
  v19 = *((_QWORD *)v13 + 7);
  v20 = v13 + 12;
  if ( !v19 )
    goto LABEL_14;
  v11 = (unsigned int)dword_50560A8;
  v21 = v13 + 12;
  do
  {
    while ( 1 )
    {
      v22 = *(_QWORD *)(v19 + 16);
      v23 = *(_QWORD *)(v19 + 24);
      if ( *(_DWORD *)(v19 + 32) >= dword_50560A8 )
        break;
      v19 = *(_QWORD *)(v19 + 24);
      if ( !v23 )
        goto LABEL_26;
    }
    v21 = (_DWORD *)v19;
    v19 = *(_QWORD *)(v19 + 16);
  }
  while ( v22 );
LABEL_26:
  if ( v20 == v21 || dword_50560A8 < v21[8] || (result = (unsigned __int8)byte_5056140, !v21[9]) )
  {
LABEL_14:
    v16 = *(__int64 (**)())(*(_QWORD *)v10 + 152LL);
    result = 0;
    if ( v16 != sub_2163C90 )
      result = ((__int64 (__fastcall *)(__int64, unsigned __int64))v16)(v10, v11);
  }
  *(_BYTE *)(a1 + 376) = result;
  return result;
}
