// Function: sub_2167560
// Address: 0x2167560
//
_BOOL8 __fastcall sub_2167560(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  int v5; // eax
  __int64 v6; // r15
  int v7; // edx
  _QWORD *v8; // rax
  __int64 (*v9)(); // r14
  __int64 **v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 (*v15)(); // rax
  __int64 (*v16)(); // rcx
  _QWORD **v17; // rsi
  __int64 v19; // r14
  __int64 v20; // r15
  unsigned __int8 v21; // r12
  unsigned __int8 v22; // r14
  __int64 v23; // rax
  unsigned int v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v5 = *((unsigned __int8 *)a2 + 16);
  v6 = *(_QWORD *)(a1 + 24);
  if ( v5 == 62 )
  {
    v8 = *(_QWORD **)v6;
    goto LABEL_15;
  }
  v7 = v5 - 24;
  v8 = *(_QWORD **)v6;
  if ( v7 != 44 )
  {
    v16 = (__int64 (*)())v8[102];
    if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
      v17 = (_QWORD **)*(a2 - 1);
    else
      v17 = (_QWORD **)&a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
    if ( v16 != sub_1D5A400 )
    {
      if ( ((unsigned __int8 (__fastcall *)(_QWORD, _QWORD, _QWORD))v16)(*(_QWORD *)(a1 + 24), **v17, *a2) )
        return 0;
      goto LABEL_6;
    }
LABEL_15:
    v15 = (__int64 (*)())v8[122];
    if ( v15 == sub_1D5A420 )
      goto LABEL_16;
    goto LABEL_18;
  }
  v9 = (__int64 (*)())v8[110];
  if ( (*((_BYTE *)a2 + 23) & 0x40) != 0 )
    v10 = (__int64 **)*(a2 - 1);
  else
    v10 = (__int64 **)&a2[-3 * (*((_DWORD *)a2 + 5) & 0xFFFFFFF)];
  LOBYTE(v11) = sub_1F59570(**v10);
  v25 = v12;
  v24 = v11;
  LOBYTE(v13) = sub_1F59570(*a2);
  if ( v9 != sub_1D5A410
    && ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v9)(v6, v13, v14, v24, v25) )
  {
    return 0;
  }
LABEL_6:
  v15 = *(__int64 (**)())(*(_QWORD *)v6 + 976LL);
  if ( v15 == sub_1D5A420 )
  {
    if ( (unsigned __int8)(*((_BYTE *)a2 + 16) - 61) <= 1u )
      goto LABEL_16;
    return 1;
  }
LABEL_18:
  if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD *))v15)(v6, a2) )
    return 0;
  if ( (unsigned __int8)(*((_BYTE *)a2 + 16) - 61) > 1u )
    return 1;
LABEL_16:
  if ( *((_BYTE *)a3 + 16) != 54 )
    return 1;
  v19 = *(_QWORD *)(a1 + 8);
  v20 = *(_QWORD *)(a1 + 24);
  v21 = sub_2167220(v19, *a2);
  v22 = sub_2167220(v19, *a3);
  v23 = a3[1];
  if ( (!v23 || *(_QWORD *)(v23 + 8))
    && (v22 && *(_QWORD *)(v20 + 8LL * v22 + 120) || !v21 || !*(_QWORD *)(v20 + 8LL * v21 + 120))
    && !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v20 + 784LL))(v20, *a2, *a3) )
  {
    return 1;
  }
  if ( !v21 || !v22 )
    return 1;
  return (((int)*(unsigned __int16 *)(v20 + 2 * (v22 + 115LL * v21 + 16104)) >> (4 * ((*((_BYTE *)a2 + 16) == 61) + 2)))
        & 0xF) != 0;
}
