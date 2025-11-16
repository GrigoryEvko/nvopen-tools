// Function: sub_33DC2B0
// Address: 0x33dc2b0
//
__int64 __fastcall sub_33DC2B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, unsigned int a6)
{
  __int64 v8; // rax
  unsigned __int16 v9; // dx
  __int64 v10; // rax
  unsigned int v11; // r12d
  _QWORD *v12; // rax
  __int64 *v13; // rax
  unsigned int v14; // r15d
  unsigned int v15; // ebx
  __int64 *v16; // r12
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  unsigned __int16 v23; // [rsp+30h] [rbp-50h] BYREF
  __int64 v24; // [rsp+38h] [rbp-48h]
  unsigned __int64 v25; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+48h] [rbp-38h]

  v8 = *(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4;
  v9 = *(_WORD *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v23 = v9;
  v24 = v10;
  if ( v9 )
  {
    if ( (unsigned __int16)(v9 - 176) > 0x34u )
      goto LABEL_3;
  }
  else if ( !sub_3007100((__int64)&v23) )
  {
    goto LABEL_6;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !v23 )
  {
LABEL_6:
    v11 = sub_3007130((__int64)&v23, a2);
    v12 = (_QWORD *)a1;
    *(_DWORD *)(a1 + 8) = v11;
    if ( v11 > 0x40 )
      goto LABEL_4;
    goto LABEL_7;
  }
  if ( (unsigned __int16)(v23 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_3:
  v11 = word_4456340[v23 - 1];
  v12 = (_QWORD *)a1;
  *(_DWORD *)(a1 + 8) = v11;
  if ( v11 > 0x40 )
  {
LABEL_4:
    sub_C43690(a1, 0, 0);
LABEL_8:
    v13 = a5;
    v14 = 0;
    v15 = v11;
    v16 = v13;
    while ( 1 )
    {
      v17 = *v16;
      v18 = 1LL << v14;
      if ( *((_DWORD *)v16 + 2) > 0x40u )
        v17 = *(_QWORD *)(v17 + 8LL * (v14 >> 6));
      if ( (v17 & v18) == 0 )
        goto LABEL_15;
      v26 = v15;
      if ( v15 <= 0x40 )
      {
        v25 = 0;
      }
      else
      {
        sub_C43690((__int64)&v25, 0, 0);
        if ( v26 > 0x40 )
        {
          *(_QWORD *)(v25 + 8LL * (v14 >> 6)) |= v18;
          goto LABEL_11;
        }
      }
      v25 |= v18;
LABEL_11:
      if ( (unsigned __int8)sub_33DC1F0(a2, a3, a4, (__int64)&v25, a6) )
      {
        v19 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a1 + 8) > 0x40u )
          *(_QWORD *)(v19 + 8LL * (v14 >> 6)) |= v18;
        else
          *(_QWORD *)a1 = v18 | v19;
      }
      if ( v26 > 0x40 && v25 )
        j_j___libc_free_0_0(v25);
LABEL_15:
      if ( ++v14 == v15 )
        return a1;
    }
  }
LABEL_7:
  *v12 = 0;
  if ( v11 )
    goto LABEL_8;
  return a1;
}
