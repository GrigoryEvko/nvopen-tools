// Function: sub_7A8210
// Address: 0x7a8210
//
__int64 __fastcall sub_7A8210(__int64 *a1, __int64 a2, unsigned __int64 a3, int a4, char a5, unsigned int a6)
{
  __int64 v8; // r12
  char v12; // al
  unsigned __int64 v13; // r14
  char v14; // al
  __int64 v15; // r15
  __int64 v16; // rdi
  __int64 i; // rax
  __int64 v18; // rax
  __int64 *v19; // r12
  char v20; // al
  _QWORD *v21; // rax
  __int64 v22; // rsi
  char v23; // al
  __int64 v24; // r12
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-78h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  char v28; // [rsp+27h] [rbp-59h]
  __int64 v29; // [rsp+28h] [rbp-58h]
  unsigned int v30; // [rsp+30h] [rbp-50h]
  __int64 v31; // [rsp+30h] [rbp-50h]
  __int64 v32; // [rsp+38h] [rbp-48h]
  unsigned __int64 v35; // [rsp+48h] [rbp-38h]

  v8 = a2;
  v29 = *a1;
  if ( !(unsigned int)sub_8D3410(a2) )
    goto LABEL_4;
  if ( dword_4D0425C )
  {
    v25 = sub_7A6790(a2);
    if ( (unsigned int)sub_8D3410(v25) )
      return 0;
    if ( (unsigned int)sub_8D43F0(a2) )
      goto LABEL_4;
  }
  else if ( (unsigned int)sub_8D43F0(a2) )
  {
    goto LABEL_4;
  }
  if ( !(unsigned int)sub_8D23B0(a2) )
  {
    v35 = sub_8D4490(a2);
    v8 = sub_8D40F0(a2);
    goto LABEL_5;
  }
LABEL_4:
  v35 = 1;
LABEL_5:
  if ( !(unsigned int)sub_8D3A70(v8) )
    return 0;
  while ( *(_BYTE *)(v8 + 140) == 12 )
    v8 = *(_QWORD *)(v8 + 160);
  v32 = *(_QWORD *)(v8 + 128);
  if ( !v35 )
    return 0;
  v12 = a5 & 1;
  v13 = 1;
  v28 = v12;
  while ( 1 )
  {
    if ( a1[7] < a3 )
      return 0;
    if ( (unsigned int)sub_7A80B0(v8) && (unsigned int)sub_7A6990(v8, v29, 0, a3, 1, 1u) )
      return 1;
    if ( a4 && **(_QWORD **)(v8 + 168) )
      break;
LABEL_15:
    if ( !a6 )
    {
      v30 = 0;
      v14 = v35 <= v13;
      goto LABEL_17;
    }
    v15 = *(_QWORD *)(v8 + 160);
    v30 = 0;
    if ( v15 )
    {
      while ( 1 )
      {
        while ( (*(_BYTE *)(v15 + 144) & 0x40) != 0 )
        {
LABEL_30:
          v15 = *(_QWORD *)(v15 + 112);
          if ( !v15 )
          {
            v14 = v30 & 1 | (v35 <= v13);
            goto LABEL_17;
          }
        }
        if ( (unsigned int)sub_8D3410(*(_QWORD *)(v15 + 120)) )
        {
          if ( (unsigned int)sub_8D43F0(*(_QWORD *)(v15 + 120)) )
          {
            v27 = 1;
          }
          else
          {
            v16 = *(_QWORD *)(v15 + 120);
            for ( i = v16; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
            if ( !*(_QWORD *)(i + 128) )
              goto LABEL_30;
            if ( dword_4D0425C )
            {
              v18 = sub_7A6790(v16);
              if ( (unsigned int)sub_8D3410(v18) )
                goto LABEL_30;
              v16 = *(_QWORD *)(v15 + 120);
            }
            v27 = sub_8D4490(v16);
          }
          v22 = sub_8D40F0(*(_QWORD *)(v15 + 120));
        }
        else
        {
          v27 = 1;
          v22 = *(_QWORD *)(v15 + 120);
        }
        while ( 1 )
        {
          v23 = *(_BYTE *)(v22 + 140);
          if ( v23 != 12 )
            break;
          v22 = *(_QWORD *)(v22 + 160);
        }
        if ( (unsigned __int8)(v23 - 9) > 2u
          || !*(_QWORD *)v22
          || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v22 + 96LL) + 180LL) & 0x40) == 0
          || !v27 )
        {
          goto LABEL_30;
        }
        v26 = v8;
        v24 = 0;
        while ( !(unsigned int)sub_7A8210(a1, v22, v24 * *(_QWORD *)(v22 + 128) + a3 + *(_QWORD *)(v15 + 128), 1, 1, 1) )
        {
          if ( v27 == ++v24 )
          {
            v8 = v26;
            goto LABEL_30;
          }
        }
        v15 = *(_QWORD *)(v15 + 112);
        v8 = v26;
        if ( !v15 )
          return 1;
        v30 = a6;
      }
    }
    v14 = v35 <= v13;
LABEL_17:
    a3 += v32;
    ++v13;
    if ( v14 )
      return v30;
  }
  v31 = v8;
  v19 = **(__int64 ***)(v8 + 168);
  while ( 1 )
  {
    v20 = *((_BYTE *)v19 + 96);
    if ( (v20 & 1) != 0 || (v20 & 2) != 0 && v28 )
    {
      if ( (unsigned int)sub_7A8210(a1, v19[5], a3 + v19[13], 1, 0, 1) )
      {
        if ( !dword_4D0425C || (v19[12] & 1) != 0 )
          return 1;
        v21 = (_QWORD *)v19[14];
        if ( v21 )
          break;
      }
    }
LABEL_36:
    v19 = (__int64 *)*v19;
    if ( !v19 )
    {
      v8 = v31;
      goto LABEL_15;
    }
  }
  while ( (*(_BYTE *)(*(_QWORD *)(v21[1] + 16LL) + 96LL) & 2) != 0 )
  {
    v21 = (_QWORD *)*v21;
    if ( !v21 )
      goto LABEL_36;
  }
  return 1;
}
