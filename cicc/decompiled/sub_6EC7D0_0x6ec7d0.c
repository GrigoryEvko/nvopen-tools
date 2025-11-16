// Function: sub_6EC7D0
// Address: 0x6ec7d0
//
__int64 *__fastcall sub_6EC7D0(_QWORD *a1, __int64 a2, _DWORD *a3, unsigned int a4)
{
  char v5; // al
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 *result; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // r13
  _QWORD *v17; // rax
  _QWORD *v18; // r14
  __int64 *v19; // rax
  __int64 v20; // r14
  __int64 *v21; // r15
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rax

  *a3 = 0;
  v5 = *((_BYTE *)a1 + 24);
  if ( v5 == 5 )
  {
    v14 = a1[7];
    if ( (*(_BYTE *)(v14 + 51) & 8) != 0 )
    {
      result = (__int64 *)sub_726700(18);
      result[7] = v14;
      *result = *a1;
      *a3 = 1;
      return result;
    }
  }
  else if ( v5 == 1 && *((_BYTE *)a1 + 56) == 3 )
  {
    v15 = a1[9];
    if ( *(_BYTE *)(v15 + 24) == 5 )
    {
      v16 = *(_QWORD *)(v15 + 56);
      if ( (*(_BYTE *)(v16 + 51) & 8) != 0 )
      {
        v17 = (_QWORD *)sub_726700(18);
        v17[7] = v16;
        v18 = v17;
        *v17 = sub_72D2E0(*a1, 0);
        result = (__int64 *)sub_73DCD0(v18);
        *a3 = 1;
        return result;
      }
    }
  }
  if ( (unsigned int)sub_731920(a1, a2, a4) )
    return (__int64 *)sub_73B8B0(a1, (*(_BYTE *)(qword_4D03C50 + 18LL) >> 2) & 4);
  v7 = *a1;
  v8 = sub_730FF0(a1, a2, v6);
  v9 = v8;
  if ( (*((_BYTE *)a1 + 25) & 1) != 0 )
  {
    v19 = (__int64 *)sub_73E1B0(v8, a2);
    v20 = *v19;
    v21 = v19;
    v22 = sub_6EAFA0(3u);
    *(_BYTE *)(v22 + 51) |= 8u;
    v23 = v22;
    *(_QWORD *)(v22 + 56) = v21;
    v24 = sub_6EC670(v20, v22, 0, 0);
    v25 = sub_73DCD0(v24);
    sub_730620(a1, v25);
    v26 = (__int64 *)sub_726700(18);
    v26[7] = v23;
    *v26 = v20;
    result = (__int64 *)sub_73DCD0(v26);
  }
  else
  {
    v10 = sub_6EAFA0(3u);
    *(_BYTE *)(v10 + 51) |= 8u;
    v11 = v10;
    *(_QWORD *)(v10 + 56) = v9;
    v12 = sub_6EC670(v7, v10, 0, 0);
    sub_730620(a1, v12);
    result = (__int64 *)sub_726700(18);
    result[7] = v11;
    *result = v7;
  }
  *a3 = 1;
  return result;
}
