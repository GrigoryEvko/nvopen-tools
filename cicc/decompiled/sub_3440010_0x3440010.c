// Function: sub_3440010
// Address: 0x3440010
//
__int64 __fastcall sub_3440010(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, __int16 a5)
{
  __int64 *v6; // rax
  __int64 *v8; // r14
  unsigned __int16 ***v10; // rbx
  unsigned __int16 *v11; // rcx
  unsigned __int16 *v12; // r12
  unsigned __int16 *v13; // r14
  __int64 v14; // rsi
  const char *(__fastcall *v15)(__int64, unsigned int); // rdx
  size_t v16; // rdx
  const char *v17; // rsi
  __int16 *v18; // rdx
  __int16 v19; // ax
  unsigned __int16 v20; // [rsp+4h] [rbp-6Ch]
  __int64 *v21; // [rsp+8h] [rbp-68h]
  unsigned __int16 ***v22; // [rsp+18h] [rbp-58h]
  __int64 *v23; // [rsp+20h] [rbp-50h]
  _BYTE *v24; // [rsp+30h] [rbp-40h] BYREF
  __int64 v25; // [rsp+38h] [rbp-38h]

  if ( !a4 || *a3 != 123 )
    return 0;
  v6 = *(__int64 **)(a2 + 288);
  v8 = *(__int64 **)(a2 + 280);
  v24 = a3 + 1;
  v25 = a4 - 2;
  v21 = v6;
  if ( v8 == v6 )
    return 0;
  v23 = v8;
  v22 = 0;
  v20 = 0;
  while ( 1 )
  {
    v10 = (unsigned __int16 ***)*v23;
    if ( (unsigned __int8)sub_2FE7BB0(a1, a2, *v23) )
    {
      v11 = **v10;
      v12 = &v11[*((unsigned __int16 *)*v10 + 10)];
      if ( v12 != v11 )
        break;
    }
LABEL_6:
    if ( v21 == ++v23 )
      return v20;
  }
  v13 = **v10;
  while ( 1 )
  {
    v14 = *v13;
    v15 = *(const char *(__fastcall **)(__int64, unsigned int))(*(_QWORD *)a2 + 632LL);
    if ( v15 == sub_2FF5340 )
    {
      v16 = 0;
      v17 = (const char *)(*(_QWORD *)(a2 + 72) + *(unsigned int *)(*(_QWORD *)(a2 + 8) + 24 * v14));
      if ( v17 )
        v16 = strlen(v17);
    }
    else
    {
      v17 = v15(a2, v14);
    }
    if ( v16 != v25 || (unsigned int)sub_C92E90(&v24, (__int64)v17, v16) )
      goto LABEL_10;
    v18 = (__int16 *)(*(_QWORD *)(a2 + 320)
                    + 2LL
                    * *(unsigned int *)(*(_QWORD *)(a2 + 312)
                                      + 16LL
                                      * (*((unsigned __int16 *)*v10 + 12)
                                       + *(_DWORD *)(a2 + 328)
                                       * (unsigned int)((__int64)(*(_QWORD *)(a2 + 288) - *(_QWORD *)(a2 + 280)) >> 3))
                                      + 12));
    v19 = *v18;
    if ( *v18 != 1 )
      break;
LABEL_21:
    if ( !v22 )
    {
      v22 = v10;
      v20 = *v13;
    }
LABEL_10:
    if ( v12 == ++v13 )
      goto LABEL_6;
  }
  while ( a5 != v19 )
  {
    v19 = v18[1];
    ++v18;
    if ( v19 == 1 )
      goto LABEL_21;
  }
  return *v13;
}
