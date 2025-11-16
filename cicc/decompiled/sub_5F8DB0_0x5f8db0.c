// Function: sub_5F8DB0
// Address: 0x5f8db0
//
__int64 *__fastcall sub_5F8DB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  char v3; // al
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r12
  __int64 **v7; // r12
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 *result; // rax
  int v17; // eax
  __int64 v18; // rdx
  char i; // al
  _BYTE *v20; // rax
  __int64 v21; // [rsp+10h] [rbp-70h]
  char v22; // [rsp+2Fh] [rbp-51h]
  __int64 v23; // [rsp+30h] [rbp-50h]
  __int64 *v24; // [rsp+38h] [rbp-48h]
  int v25; // [rsp+48h] [rbp-38h] BYREF
  int v26; // [rsp+4Ch] [rbp-34h] BYREF

  v2 = a1;
  v3 = *(_BYTE *)(a1 + 174);
  v4 = *(_QWORD *)(a1 + 152);
  v25 = 0;
  v22 = v3;
  v24 = *(__int64 **)(v4 + 168);
  if ( (*(_BYTE *)(a1 + 194) & 0x40) == 0 )
  {
    v5 = 0;
    v23 = *v24;
    if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
      goto LABEL_3;
LABEL_29:
    v6 = *(_QWORD *)(v23 + 8);
    for ( i = *(_BYTE *)(v6 + 140); i == 12; i = *(_BYTE *)(v6 + 140) )
      v6 = *(_QWORD *)(v6 + 160);
    if ( i == 6 )
    {
      do
        v6 = *(_QWORD *)(v6 + 160);
      while ( *(_BYTE *)(v6 + 140) == 12 );
    }
    goto LABEL_4;
  }
  v23 = 0;
  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 360) + 24LL) + 40LL);
  if ( (*(_BYTE *)(a1 + 89) & 4) == 0 )
    goto LABEL_29;
LABEL_3:
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
LABEL_4:
  if ( !**(_QWORD **)(v6 + 168) )
    goto LABEL_22;
  v21 = v6;
  v7 = **(__int64 ****)(v6 + 168);
  do
  {
    if ( (unsigned int)sub_8DBE70(v7[5]) )
    {
LABEL_15:
      v25 = 1;
      goto LABEL_16;
    }
    if ( ((_BYTE)v7[12] & 1) != 0 )
    {
      v11 = (__int64)v7[5];
      if ( v5 )
      {
        if ( v5 == v11 || (unsigned int)sub_8D97D0(v7[5], v5, 0, v9, v10) )
        {
          v17 = sub_5F8900(a2, v4, v8, v9, v10);
          v25 = v17;
          goto LABEL_19;
        }
        v11 = (__int64)v7[5];
      }
      v26 = 0;
      if ( (unsigned int)sub_8D23B0(v11) || (unsigned __int8)(*(_BYTE *)(v11 + 140) - 9) > 2u )
      {
        if ( v26 )
          goto LABEL_15;
      }
      else
      {
        v12 = sub_5E8390(v11, v22, v23, 0, (int)a1 + 64, (int)&v26);
        if ( v26 )
          goto LABEL_15;
        if ( v12 )
        {
          v17 = sub_5F8900(v12, v4, v13, v14, v15);
          v25 = v17;
LABEL_19:
          if ( v17 )
            goto LABEL_16;
          goto LABEL_20;
        }
      }
      v17 = v25;
      goto LABEL_19;
    }
LABEL_20:
    v7 = (__int64 **)*v7;
  }
  while ( v7 );
  v6 = v21;
  v2 = a1;
  if ( !v25 )
  {
LABEL_22:
    sub_5F8AD0(v2, v6, *(__int64 **)(v6 + 160), &v25);
    if ( !v25 )
    {
      result = (__int64 *)v24[7];
      if ( result )
        goto LABEL_24;
      v20 = (_BYTE *)sub_725E60(v2, v6, v18);
      v24[7] = (__int64)v20;
      *v20 |= 8u;
    }
  }
LABEL_16:
  result = (__int64 *)v24[7];
  if ( result )
  {
LABEL_24:
    *(_BYTE *)result |= 8u;
    if ( v25 )
    {
      v24[7] = 0;
      return v24;
    }
    else
    {
      result = (__int64 *)dword_4D048B4;
      if ( dword_4D048B4 )
      {
        result = (__int64 *)v24[7];
        if ( !result[1] )
        {
          *(_BYTE *)result |= 1u;
          result = (__int64 *)v24[7];
          result[1] = 0;
        }
      }
    }
  }
  return result;
}
