// Function: sub_32886A0
// Address: 0x32886a0
//
__int64 __fastcall sub_32886A0(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, int a6)
{
  bool v6; // zf
  __int64 v10; // r8
  __int64 v11; // r9
  _BYTE *v12; // rcx
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // r12
  __int64 v17; // rax
  _BYTE *v18; // rdx
  __int128 v19; // [rsp-10h] [rbp-170h]
  int v20; // [rsp+4h] [rbp-15Ch]
  __int64 v21; // [rsp+8h] [rbp-158h]
  __int64 v22; // [rsp+10h] [rbp-150h] BYREF
  __int64 v23; // [rsp+18h] [rbp-148h]
  _BYTE *v24; // [rsp+20h] [rbp-140h] BYREF
  __int64 v25; // [rsp+28h] [rbp-138h]
  _BYTE v26[304]; // [rsp+30h] [rbp-130h] BYREF

  v6 = *(_DWORD *)(a5 + 24) == 51;
  v22 = a2;
  v23 = a3;
  if ( !v6 )
  {
    if ( (_WORD)v22 )
    {
      if ( (unsigned __int16)(v22 - 176) > 0x34u )
        goto LABEL_15;
    }
    else if ( !sub_3007100((__int64)&v22) )
    {
LABEL_4:
      v10 = (unsigned int)sub_3007130((__int64)&v22, a2);
LABEL_5:
      v11 = (unsigned int)v10;
      v24 = v26;
      v25 = 0x1000000000LL;
      if ( (unsigned int)v10 > 0x10 )
      {
        v20 = v10;
        v21 = (unsigned int)v10;
        sub_C8D5F0((__int64)&v24, v26, (unsigned int)v10, 0x10u, v10, (unsigned int)v10);
        v11 = v21;
        v17 = (__int64)v24;
        v18 = &v24[16 * v21];
        do
        {
          if ( v17 )
          {
            *(_QWORD *)v17 = a5;
            *(_DWORD *)(v17 + 8) = a6;
          }
          v17 += 16;
        }
        while ( v18 != (_BYTE *)v17 );
        LODWORD(v25) = v20;
        v12 = v24;
      }
      else
      {
        v12 = v26;
        if ( (_DWORD)v10 )
        {
          v13 = (unsigned int)v10;
          v14 = v26;
          do
          {
            *v14 = a5;
            v14 += 2;
            *((_DWORD *)v14 - 2) = a6;
            --v13;
          }
          while ( v13 );
          v12 = v24;
        }
        LODWORD(v25) = v10;
      }
      *((_QWORD *)&v19 + 1) = v11;
      *(_QWORD *)&v19 = v12;
      v15 = sub_33FC220(a1, 156, a4, v22, v23, v11, v19);
      if ( v24 != v26 )
        _libc_free((unsigned __int64)v24);
      return v15;
    }
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    if ( !(_WORD)v22 )
      goto LABEL_4;
    if ( (unsigned __int16)(v22 - 176) <= 0x34u )
      sub_CA17B0(
        "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT"
        "::getVectorElementCount() instead");
LABEL_15:
    v10 = word_4456340[(unsigned __int16)v22 - 1];
    goto LABEL_5;
  }
  v24 = 0;
  LODWORD(v25) = 0;
  v15 = sub_33F17F0(a1, 51, &v24, (unsigned int)v22, a3);
  if ( v24 )
    sub_B91220((__int64)&v24, (__int64)v24);
  return v15;
}
