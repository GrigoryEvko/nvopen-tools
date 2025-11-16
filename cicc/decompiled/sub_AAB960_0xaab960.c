// Function: sub_AAB960
// Address: 0xaab960
//
__int64 __fastcall sub_AAB960(__int64 a1, unsigned __int8 *a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v7; // r14
  _QWORD *v10; // rbx
  int v11; // eax
  _QWORD *v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rsi
  int v15; // edx
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  _QWORD *v22; // r13
  unsigned __int8 *v23; // r15
  unsigned __int8 *v24; // r15
  unsigned __int8 *v25; // r15
  unsigned __int8 *v26; // r15
  int v27; // edx
  unsigned __int8 *v28; // r13
  unsigned __int8 *v29; // r13
  unsigned __int8 *v30; // r13
  _QWORD *v31; // [rsp+8h] [rbp-48h]
  _QWORD *v32; // [rsp+8h] [rbp-48h]
  size_t v33; // [rsp+18h] [rbp-38h]

  if ( !a5 )
    return (__int64)a2;
  v7 = *((_QWORD *)a2 + 1);
  v10 = a4;
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
  {
    v12 = &a4[a5];
    if ( a4 != v12 )
    {
      v13 = a4;
      v14 = *(_QWORD *)(*a4 + 8LL);
      v15 = *(unsigned __int8 *)(v14 + 8);
      if ( v15 == 17 )
      {
LABEL_12:
        v16 = 0;
      }
      else
      {
        while ( v15 != 18 )
        {
          if ( v12 == ++v13 )
            goto LABEL_4;
          v14 = *(_QWORD *)(*v13 + 8LL);
          v15 = *(unsigned __int8 *)(v14 + 8);
          if ( v15 == 17 )
            goto LABEL_12;
        }
        v16 = 1;
      }
      BYTE4(v33) = v16;
      v31 = a4;
      LODWORD(v33) = *(_DWORD *)(v14 + 32);
      v17 = sub_BCE1B0(v7, v33);
      a4 = v31;
      v7 = v17;
    }
  }
LABEL_4:
  v11 = *a2;
  if ( (_BYTE)v11 == 13 )
    return sub_ACADE0(v7);
  if ( (unsigned int)(v11 - 12) <= 1 )
    return sub_ACA8A0(v7);
  if ( *(_BYTE *)(a3 + 32) )
    return 0;
  v18 = 8 * a5;
  v32 = &a4[(unsigned __int64)v18 / 8];
  v19 = v18;
  v20 = v18 >> 5;
  v21 = v19 >> 3;
  if ( v20 > 0 )
  {
    v22 = &a4[4 * v20];
    while ( 1 )
    {
      v26 = (unsigned __int8 *)*v10;
      if ( !(unsigned __int8)sub_AC30F0(*v10) && (unsigned int)*v26 - 12 > 1 )
        goto LABEL_27;
      v23 = (unsigned __int8 *)v10[1];
      if ( !(unsigned __int8)sub_AC30F0(v23) && (unsigned int)*v23 - 12 > 1 )
      {
        ++v10;
        goto LABEL_27;
      }
      v24 = (unsigned __int8 *)v10[2];
      if ( !(unsigned __int8)sub_AC30F0(v24) && (unsigned int)*v24 - 12 > 1 )
      {
        v10 += 2;
        goto LABEL_27;
      }
      v25 = (unsigned __int8 *)v10[3];
      if ( !(unsigned __int8)sub_AC30F0(v25) && (unsigned int)*v25 - 12 > 1 )
      {
        v10 += 3;
        goto LABEL_27;
      }
      v10 += 4;
      if ( v22 == v10 )
      {
        v21 = v32 - v10;
        break;
      }
    }
  }
  switch ( v21 )
  {
    case 2LL:
      goto LABEL_46;
    case 3LL:
      v29 = (unsigned __int8 *)*v10;
      if ( !(unsigned __int8)sub_AC30F0(*v10) && (unsigned int)*v29 - 12 > 1 )
        goto LABEL_27;
      ++v10;
LABEL_46:
      v30 = (unsigned __int8 *)*v10;
      if ( !(unsigned __int8)sub_AC30F0(*v10) && (unsigned int)*v30 - 12 > 1 )
        goto LABEL_27;
      ++v10;
      goto LABEL_40;
    case 1LL:
LABEL_40:
      v28 = (unsigned __int8 *)*v10;
      if ( (unsigned __int8)sub_AC30F0(*v10) || (unsigned int)*v28 - 12 <= 1 )
        break;
LABEL_27:
      if ( v32 == v10 )
        break;
      return 0;
  }
  v27 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v27 - 17) > 1 || (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a2 + 1) + 8LL) - 17 <= 1 )
    return (__int64)a2;
  BYTE4(v33) = (_BYTE)v27 == 18;
  LODWORD(v33) = *(_DWORD *)(v7 + 32);
  return sub_AD5E10(v33);
}
