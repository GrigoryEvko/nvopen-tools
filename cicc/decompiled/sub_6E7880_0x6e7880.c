// Function: sub_6E7880
// Address: 0x6e7880
//
_DWORD *__fastcall sub_6E7880(__int64 a1, _DWORD *a2, int a3, __int64 a4, __int64 *a5, _DWORD *a6, _DWORD *a7)
{
  _DWORD *result; // rax
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // r15
  __int64 i; // r14
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  _QWORD *v16; // r12
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r12
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 v32; // rax
  _QWORD *v33; // [rsp+0h] [rbp-40h]

  result = a7;
  if ( !a7 )
  {
    v9 = *((_BYTE *)a2 + 96);
    if ( (v9 & 4) != 0 && a3 )
    {
      v30 = *((_QWORD *)a2 + 5);
      a2 = a6;
      v31 = a1;
      a1 = 287;
      sub_6E5ED0(0x11Fu, a6, v31, v30);
    }
    else
    {
      if ( (v9 & 2) == 0 )
      {
        v10 = *((_QWORD *)a2 + 14);
        v11 = *(_QWORD *)(v10 + 8);
        result = *(_DWORD **)(v11 + 16);
        if ( (result[24] & 2) == 0 )
          goto LABEL_12;
      }
      v28 = *((_QWORD *)a2 + 5);
      a2 = a6;
      v29 = a1;
      a1 = 288;
      sub_6E5ED0(0x120u, a6, v29, v28);
    }
LABEL_6:
    result = (_DWORD *)sub_7305B0(a1, a2);
    *a5 = (__int64)result;
    return result;
  }
  *a7 = 0;
  if ( (a2[24] & 4) != 0 && a3
    || (a2[24] & 2) != 0
    || (v10 = *((_QWORD *)a2 + 14), v11 = *(_QWORD *)(v10 + 8), (*(_BYTE *)(*(_QWORD *)(v11 + 16) + 96LL) & 2) != 0) )
  {
    *a7 = 1;
    goto LABEL_6;
  }
LABEL_12:
  for ( i = *(_QWORD *)(v10 + 16); *(_QWORD *)(v11 + 8) != i; i = *(_QWORD *)(i + 8) )
  {
    if ( i == v11 )
    {
      if ( (unsigned int)sub_8D2EF0(*(_QWORD *)*a5) )
      {
        v32 = sub_72D740(a1, *(_QWORD *)*a5, v13, v14, v15);
        v16 = (_QWORD *)*a5;
        v17 = v32;
      }
      else
      {
        v16 = (_QWORD *)*a5;
        v17 = a1;
      }
      v18 = sub_73DBF0(15, v17, *a5);
      *a5 = v18;
      *(_QWORD *)(v18 + 28) = *(_QWORD *)a6;
      result = (_DWORD *)sub_730580(v16, v18);
    }
    else
    {
      v19 = 0;
      v20 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(i + 8) + 16LL) + 40LL);
      if ( (*(_BYTE *)(a1 + 140) & 0xFB) == 8 )
        v19 = (unsigned int)sub_8D4C10(a1, dword_4F077C4 != 2);
      while ( *(_BYTE *)(v20 + 140) == 12 )
        v20 = *(_QWORD *)(v20 + 160);
      v21 = sub_73C570(v20, v19, -1);
      if ( (unsigned int)sub_8D2EF0(*(_QWORD *)*a5) )
        v25 = sub_72D740(v21, *(_QWORD *)*a5, v22, v23, v24);
      else
        v25 = v21;
      v33 = (_QWORD *)*a5;
      v26 = sub_73DBF0(15, v25, *a5);
      *a5 = v26;
      v27 = v26;
      *(_QWORD *)(v26 + 28) = *(_QWORD *)a6;
      result = (_DWORD *)sub_730580(v33, v26);
      *(_BYTE *)(v27 + 58) |= 0x80u;
    }
  }
  return result;
}
