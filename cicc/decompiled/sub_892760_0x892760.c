// Function: sub_892760
// Address: 0x892760
//
_BOOL8 __fastcall sub_892760(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r9
  int v6; // esi
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 *v10; // rax
  _QWORD *v11; // r13
  _BOOL8 result; // rax
  char v13; // bl
  __int64 v14; // rax
  _BOOL8 v15; // [rsp-10h] [rbp-50h]
  __int64 v17; // [rsp+10h] [rbp-30h]
  __int64 v18; // [rsp+18h] [rbp-28h]

  v4 = 0;
  v6 = a4;
  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    v4 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
  v8 = *(_QWORD *)(a3 + 88);
  v9 = *(_QWORD *)(a1 + 240);
  if ( v8 && (*(_BYTE *)(a3 + 160) & 1) == 0 )
  {
    v10 = *(__int64 **)(*(_QWORD *)(v8 + 88) + 328LL);
    if ( v10 )
      goto LABEL_6;
  }
  else
  {
    v10 = *(__int64 **)(a3 + 328);
    if ( v10 )
      goto LABEL_6;
  }
  v17 = v4;
  v18 = *(_QWORD *)(a1 + 240);
  v14 = sub_892400(a3);
  v6 = a4;
  v4 = v17;
  v10 = *(__int64 **)(v14 + 32);
  v9 = v18;
LABEL_6:
  v11 = (_QWORD *)sub_5CF220(*(const __m128i **)(*(_QWORD *)(a3 + 176) + 104LL), v6, a2, *v10, v9, v4, 0, 0);
  result = v15;
  if ( v11 )
  {
    v13 = sub_67D840(3646);
    sub_67D850(3646, 3, 1);
    sub_5CEC90(v11, a1, 11);
    return sub_67D850(3646, v13, 1);
  }
  return result;
}
