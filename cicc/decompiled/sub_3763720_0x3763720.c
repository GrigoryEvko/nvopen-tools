// Function: sub_3763720
// Address: 0x3763720
//
__int64 __fastcall sub_3763720(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 v7; // bx
  __int64 v8; // rax
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rbx
  __int64 result; // rax
  int v12; // r13d
  int v13; // r12d
  int v14; // ebx
  int v15; // r15d
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  int v19; // [rsp+8h] [rbp-68h]
  unsigned int v20; // [rsp+Ch] [rbp-64h]
  __int64 v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 v22; // [rsp+18h] [rbp-58h]
  unsigned __int16 v23; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+28h] [rbp-48h]

  v7 = a1;
  v21 = a1;
  v22 = a2;
  if ( (_WORD)a1 )
  {
    if ( (unsigned __int16)(a1 - 17) <= 0xD3u )
    {
      v7 = word_4456580[(unsigned __int16)a1 - 1];
      v24 = 0;
      v23 = v7;
      if ( !v7 )
        goto LABEL_5;
      goto LABEL_21;
    }
    goto LABEL_3;
  }
  if ( !sub_30070B0((__int64)&v21) )
  {
LABEL_3:
    v8 = v22;
    goto LABEL_4;
  }
  v7 = sub_3009970((__int64)&v21, a2, v16, v17, a5);
  v8 = v18;
LABEL_4:
  v23 = v7;
  v24 = v8;
  if ( !v7 )
  {
LABEL_5:
    v9 = sub_3007260((__int64)&v23);
    goto LABEL_6;
  }
LABEL_21:
  if ( v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
    BUG();
  v9 = *(_QWORD *)&byte_444C4A0[16 * v7 - 16];
LABEL_6:
  v10 = v9 >> 3;
  if ( (_WORD)v21 )
  {
    if ( (unsigned __int16)(v21 - 176) > 0x34u )
      goto LABEL_8;
  }
  else if ( !sub_3007100((__int64)&v21) )
  {
    goto LABEL_11;
  }
  sub_CA17B0(
    "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::ge"
    "tVectorElementCount() instead");
  if ( !(_WORD)v21 )
  {
LABEL_11:
    v20 = sub_3007130((__int64)&v21, a2);
    result = v20;
    if ( !v20 )
      return result;
    goto LABEL_12;
  }
  if ( (unsigned __int16)(v21 - 176) <= 0x34u )
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_8:
  v20 = word_4456340[(unsigned __int16)v21 - 1];
  result = word_4456340[(unsigned __int16)v21 - 1];
  if ( !word_4456340[(unsigned __int16)v21 - 1] )
    return result;
LABEL_12:
  result = a3 + 16;
  v19 = v10;
  v12 = v10 - 1;
  v13 = 0;
  v14 = -1;
  do
  {
    if ( v12 >= 0 )
    {
      result = *(unsigned int *)(a3 + 8);
      v15 = v14 + v12 + 1;
      do
      {
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), result + 1, 4u, a5, a6);
          result = *(unsigned int *)(a3 + 8);
        }
        *(_DWORD *)(*(_QWORD *)a3 + 4 * result) = v15--;
        result = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
        *(_DWORD *)(a3 + 8) = result;
      }
      while ( v15 != v14 );
    }
    ++v13;
    v14 += v19;
  }
  while ( v20 != v13 );
  return result;
}
