// Function: sub_2293160
// Address: 0x2293160
//
__int64 __fastcall sub_2293160(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r9
  _QWORD *v14; // r13
  _QWORD *v15; // r12
  __int64 result; // rax
  __int64 *v17; // rdi
  _QWORD *v18; // rax
  __int64 v19; // r9
  _QWORD *v20; // r13
  _QWORD *v21; // r14
  __int64 v22; // rdi
  __int64 v23; // rdi
  unsigned __int64 v24; // r14
  unsigned __int64 v25; // rax
  __int64 v26; // r12
  _BYTE *v28; // [rsp+10h] [rbp-B0h]
  _BYTE *v29; // [rsp+18h] [rbp-A8h]
  __int64 v30; // [rsp+20h] [rbp-A0h]
  unsigned __int8 v31; // [rsp+20h] [rbp-A0h]
  unsigned __int8 v33; // [rsp+28h] [rbp-98h]
  unsigned __int64 v34[2]; // [rsp+30h] [rbp-90h] BYREF
  _BYTE v35[32]; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v36[2]; // [rsp+60h] [rbp-60h] BYREF
  _BYTE v37[80]; // [rsp+70h] [rbp-50h] BYREF

  v29 = 0;
  if ( *a2 > 0x1Cu && (unsigned __int8)(*a2 - 61) <= 1u )
    v29 = (_BYTE *)*((_QWORD *)a2 - 4);
  v28 = 0;
  if ( *a3 > 0x1Cu && (unsigned __int8)(*a3 - 61) <= 1u )
    v28 = (_BYTE *)*((_QWORD *)a3 - 4);
  v10 = 0;
  v11 = sub_D97190(*(_QWORD *)(a1 + 8), a4);
  if ( *(_WORD *)(v11 + 24) == 15 )
    v10 = v11 - 32;
  v12 = sub_D97190(*(_QWORD *)(a1 + 8), a5);
  v13 = 0;
  if ( *(_WORD *)(v12 + 24) == 15 )
    v13 = v12 - 32;
  v30 = v13;
  v14 = sub_DCADF0(*(__int64 **)(a1 + 8), (__int64)a2);
  v15 = sub_DCADF0(*(__int64 **)(a1 + 8), (__int64)a3);
  if ( v15 != v14 )
    return 0;
  v17 = *(__int64 **)(a1 + 8);
  if ( v10 )
    v10 += 32;
  v18 = sub_DCC810(v17, a4, v10, 0, 0);
  v19 = v30;
  v20 = v18;
  if ( v30 )
    v19 = v30 + 32;
  v21 = sub_DCC810(*(__int64 **)(a1 + 8), a5, v19, 0, 0);
  result = 0;
  if ( *((_WORD *)v20 + 12) == 8 )
  {
    if ( *((_WORD *)v21 + 12) != 8 || v20[5] != 2 || v21[5] != 2 )
      return 0;
    v22 = *(_QWORD *)(a1 + 8);
    v34[0] = (unsigned __int64)v35;
    v34[1] = 0x400000000LL;
    sub_30B6B70(v22, v20, v34);
    sub_30B6B70(*(_QWORD *)(a1 + 8), v21, v34);
    v23 = *(_QWORD *)(a1 + 8);
    v36[0] = (unsigned __int64)v37;
    v36[1] = 0x400000000LL;
    sub_30B7CA0(v23, v34, v36, v15);
    sub_30B84A0(*(_QWORD *)(a1 + 8), v20, a6, v36);
    sub_30B84A0(*(_QWORD *)(a1 + 8), v21, a7, v36);
    v24 = *(unsigned int *)(a6 + 8);
    if ( v24 > 1 )
    {
      v25 = *(unsigned int *)(a7 + 8);
      if ( v24 == v25 && v25 > 1 )
      {
        if ( (_BYTE)qword_4FDB4C8 )
        {
LABEL_32:
          result = 1;
LABEL_33:
          if ( (_BYTE *)v36[0] != v37 )
          {
            v31 = result;
            _libc_free(v36[0]);
            result = v31;
          }
          if ( (_BYTE *)v34[0] != v35 )
          {
            v33 = result;
            _libc_free(v34[0]);
            return v33;
          }
          return result;
        }
        v26 = 1;
        while ( (unsigned __int8)sub_228E2E0(a1, *(_QWORD *)(*(_QWORD *)a6 + 8 * v26), v29)
             && (unsigned __int8)sub_228E170(
                                   a1,
                                   *(_QWORD *)(*(_QWORD *)a6 + 8 * v26),
                                   *(_QWORD *)(v36[0] + 8 * v26 - 8))
             && (unsigned __int8)sub_228E2E0(a1, *(_QWORD *)(*(_QWORD *)a7 + 8 * v26), v28)
             && (unsigned __int8)sub_228E170(
                                   a1,
                                   *(_QWORD *)(*(_QWORD *)a7 + 8 * v26),
                                   *(_QWORD *)(v36[0] + 8 * v26 - 8)) )
        {
          if ( v24 == ++v26 )
            goto LABEL_32;
        }
      }
    }
    result = 0;
    goto LABEL_33;
  }
  return result;
}
