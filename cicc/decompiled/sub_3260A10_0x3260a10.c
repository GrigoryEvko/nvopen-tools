// Function: sub_3260A10
// Address: 0x3260a10
//
char *__fastcall sub_3260A10(unsigned int *src, unsigned int *a2, unsigned int *a3, unsigned int *a4, _QWORD *a5)
{
  unsigned int *v5; // r14
  unsigned int *v6; // r13
  __int64 v8; // rax
  unsigned __int16 *v9; // rax
  int v10; // r12d
  __int64 v11; // rax
  unsigned int v12; // r12d
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  char *v16; // rbx
  unsigned __int16 v18; // dx
  unsigned __int16 v19; // [rsp+8h] [rbp-68h]
  unsigned __int16 v21; // [rsp+20h] [rbp-50h] BYREF
  __int64 v22; // [rsp+28h] [rbp-48h]
  __int16 v23; // [rsp+30h] [rbp-40h] BYREF
  __int64 v24; // [rsp+38h] [rbp-38h]

  v5 = a3;
  v6 = src;
  if ( a3 != a4 && src != a2 )
  {
    do
    {
      v9 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2]);
      v10 = *v9;
      v11 = *((_QWORD *)v9 + 1);
      v23 = v10;
      v24 = v11;
      if ( (_WORD)v10 )
      {
        if ( (unsigned __int16)(v10 - 176) <= 0x34u )
        {
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
        }
        v12 = word_4456340[v10 - 1];
      }
      else
      {
        if ( sub_3007100((__int64)&v23) )
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
        v12 = sub_3007130((__int64)&v23, (__int64)a2);
      }
      v13 = *(_QWORD *)(*(_QWORD *)v6 + 48LL) + 16LL * v6[2];
      v18 = *(_WORD *)v13;
      v14 = *(_QWORD *)(v13 + 8);
      v21 = v18;
      v22 = v14;
      if ( v18 )
      {
        if ( (unsigned __int16)(v18 - 176) <= 0x34u )
        {
          v19 = v18;
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
          sub_CA17B0(
            "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " MVT::getVectorElementCount() instead");
          v18 = v19;
        }
        if ( word_4456340[v18 - 1] < v12 )
        {
LABEL_7:
          v8 = *(_QWORD *)v5;
          a5 += 2;
          v5 += 4;
          *(a5 - 2) = v8;
          *((_DWORD *)a5 - 2) = *(v5 - 2);
          if ( v6 == a2 )
            break;
          continue;
        }
      }
      else
      {
        if ( sub_3007100((__int64)&v21) )
          sub_CA17B0(
            "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use"
            " EVT::getVectorElementCount() instead");
        if ( (unsigned int)sub_3007130((__int64)&v21, (__int64)a2) < v12 )
          goto LABEL_7;
      }
      v15 = *(_QWORD *)v6;
      a5 += 2;
      v6 += 4;
      *(a5 - 2) = v15;
      *((_DWORD *)a5 - 2) = *(v6 - 2);
      if ( v6 == a2 )
        break;
    }
    while ( v5 != a4 );
  }
  if ( a2 != v6 )
    memmove(a5, v6, (char *)a2 - (char *)v6);
  v16 = (char *)a5 + (char *)a2 - (char *)v6;
  if ( a4 != v5 )
    memmove(v16, v5, (char *)a4 - (char *)v5);
  return &v16[(char *)a4 - (char *)v5];
}
