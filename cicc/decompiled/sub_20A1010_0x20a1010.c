// Function: sub_20A1010
// Address: 0x20a1010
//
__int64 __fastcall sub_20A1010(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, char a5)
{
  __int64 *v6; // rax
  __int64 *v8; // r14
  unsigned __int16 ***v10; // rbx
  unsigned __int16 *v11; // r14
  unsigned __int16 *v12; // r12
  __int64 v13; // rsi
  const char *(__fastcall *v14)(__int64, unsigned int); // rax
  size_t v15; // rdx
  const char *v16; // rsi
  char *v17; // rdx
  char v18; // al
  unsigned __int16 v19; // [rsp+4h] [rbp-6Ch]
  __int64 *v20; // [rsp+8h] [rbp-68h]
  unsigned __int16 ***v21; // [rsp+18h] [rbp-58h]
  __int64 *v22; // [rsp+20h] [rbp-50h]
  _BYTE *v23; // [rsp+30h] [rbp-40h] BYREF
  __int64 v24; // [rsp+38h] [rbp-38h]

  if ( !a4 || *a3 != 123 )
    return 0;
  v6 = *(__int64 **)(a2 + 264);
  v8 = *(__int64 **)(a2 + 256);
  v23 = a3 + 1;
  v24 = a4 - 2;
  v20 = v6;
  if ( v6 == v8 )
    return 0;
  v22 = v8;
  v21 = 0;
  v19 = 0;
  while ( 1 )
  {
    v10 = (unsigned __int16 ***)*v22;
    if ( (unsigned __int8)sub_1F41770(a1, a2, *v22) )
    {
      v11 = **v10;
      v12 = &v11[*((unsigned __int16 *)*v10 + 10)];
      if ( v12 != v11 )
        break;
    }
LABEL_6:
    if ( v20 == ++v22 )
      return v19;
  }
  while ( 1 )
  {
    v13 = *v11;
    v14 = *(const char *(__fastcall **)(__int64, unsigned int))(*(_QWORD *)a2 + 400LL);
    if ( v14 == sub_1F49CB0 )
    {
      v15 = 0;
      v16 = (const char *)(*(_QWORD *)(a2 + 72) + *(unsigned int *)(*(_QWORD *)(a2 + 8) + 24 * v13));
      if ( v16 )
        v15 = strlen(v16);
    }
    else
    {
      v16 = v14(a2, v13);
    }
    if ( v15 != v24 || (unsigned int)sub_16D1F70(&v23, (__int64)v16, v15) )
      goto LABEL_10;
    v17 = *(char **)(*(_QWORD *)(a2 + 280)
                   + 24LL
                   * (*((unsigned __int16 *)*v10 + 12)
                    + *(_DWORD *)(a2 + 288)
                    * (unsigned int)((__int64)(*(_QWORD *)(a2 + 264) - *(_QWORD *)(a2 + 256)) >> 3))
                   + 16);
    v18 = *v17;
    if ( *v17 != 1 )
      break;
LABEL_21:
    if ( !v21 )
    {
      v21 = v10;
      v19 = *v11;
    }
LABEL_10:
    if ( ++v11 == v12 )
      goto LABEL_6;
  }
  while ( a5 != v18 )
  {
    v18 = *++v17;
    if ( v18 == 1 )
      goto LABEL_21;
  }
  return *v11;
}
