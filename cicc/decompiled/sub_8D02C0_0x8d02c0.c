// Function: sub_8D02C0
// Address: 0x8d02c0
//
__int64 *__fastcall sub_8D02C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  _QWORD *v13; // rdx
  __int64 v14; // rax
  _QWORD *v15; // rdx
  __int64 v16; // r12
  _QWORD *v17; // [rsp+18h] [rbp-48h]
  _QWORD *v18; // [rsp+18h] [rbp-48h]
  char v19; // [rsp+27h] [rbp-39h] BYREF
  __m128i *v20[7]; // [rsp+28h] [rbp-38h] BYREF

  v2 = sub_87D510(a1, &v19);
  v3 = *(_QWORD *)(v2 + 32);
  if ( !v3 )
  {
    v16 = v2;
    sub_8C7090(v19, v2);
    v3 = *(_QWORD *)(v16 + 32);
  }
  v4 = sub_878920(a1);
  v5 = sub_892920(v4);
  v6 = sub_8CFEE0(v5, a2);
  v7 = v6;
  if ( !v6 )
    return 0;
  switch ( *(_BYTE *)(v6 + 80) )
  {
    case 4:
    case 5:
      v17 = *(_QWORD **)(*(_QWORD *)(v6 + 96) + 80LL);
      break;
    case 6:
      v17 = *(_QWORD **)(*(_QWORD *)(v6 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v17 = *(_QWORD **)(*(_QWORD *)(v6 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v17 = *(_QWORD **)(v6 + 88);
      break;
    default:
      BUG();
  }
  v8 = v17[22];
  if ( !v8
    || (v9 = sub_87D1A0(v17[22], (char *)v20)) == 0
    || v3 != *(_QWORD *)(v9 + 32)
    || !(unsigned int)sub_880F30(v8, a2) )
  {
    v10 = v17[18];
    if ( v10 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(*(_QWORD *)(v10 + 88) + 176LL);
        v11 = sub_87D1A0(v8, (char *)v20);
        if ( v11 )
        {
          if ( v3 == *(_QWORD *)(v11 + 32) && (unsigned int)sub_880F30(v8, a2) )
            break;
        }
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          goto LABEL_17;
      }
      if ( v8 )
        return (__int64 *)v8;
    }
LABEL_17:
    v13 = (_QWORD *)v17[21];
    if ( !v13 )
    {
LABEL_24:
      v20[0] = sub_72F240(*(const __m128i **)(*(_QWORD *)(*(_QWORD *)(a1 + 88) + 168LL) + 168LL));
      return sub_8A0370(v7, v20, 0, 0, 0, 0, 0);
    }
    while ( 1 )
    {
      v8 = v13[1];
      v18 = v13;
      v14 = sub_87D1A0(v8, (char *)v20);
      v15 = v18;
      if ( v14 && v3 == *(_QWORD *)(v14 + 32) )
      {
        if ( (unsigned int)sub_880F30(v8, a2) )
        {
          if ( !v8 )
            goto LABEL_24;
          return (__int64 *)v8;
        }
        v15 = v18;
      }
      v13 = (_QWORD *)*v15;
      if ( !v13 )
        goto LABEL_24;
    }
  }
  return (__int64 *)v8;
}
