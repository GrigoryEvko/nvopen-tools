// Function: sub_1C62450
// Address: 0x1c62450
//
__int64 __fastcall sub_1C62450(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v7; // rcx
  int v8; // r12d
  _QWORD *v9; // r14
  unsigned __int64 v10; // r8
  __int64 v11; // rbx
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // r12d
  _QWORD *v16; // r14
  unsigned __int64 v17; // r8
  __int64 v18; // rbx
  _BYTE *v19; // rsi
  int v20; // [rsp+Ch] [rbp-84h]
  __int64 v21; // [rsp+10h] [rbp-80h]
  __int64 v23; // [rsp+20h] [rbp-70h]
  unsigned __int64 v24; // [rsp+28h] [rbp-68h]
  unsigned __int64 v25; // [rsp+28h] [rbp-68h]
  unsigned __int64 v26; // [rsp+48h] [rbp-48h] BYREF
  unsigned __int64 v27; // [rsp+50h] [rbp-40h] BYREF
  unsigned __int64 v28[7]; // [rsp+58h] [rbp-38h] BYREF

  v3 = *a2;
  result = (__int64)(*(_QWORD *)(*(_QWORD *)*a2 + 8LL) - **(_QWORD **)*a2) >> 3;
  if ( !(_DWORD)result )
    return result;
  v20 = result - 1;
  v21 = (unsigned int)result;
  v23 = 0;
  while ( 2 )
  {
    v26 = 0;
    if ( a2[1] == v3 )
      goto LABEL_15;
    v7 = 0;
    v8 = 0;
    v9 = a1;
    do
    {
      v11 = *(_QWORD *)(**(_QWORD **)(v3 + 8 * v7) + 8 * v23);
      sub_1C620D0(v9, *(unsigned int **)v11, *(__int64 ***)(v11 + 8), &v27, v28, v9[25], 0);
      v10 = v27;
      if ( *(_BYTE *)(v27 + 16) == 18 )
      {
        if ( v27 == *(_QWORD *)(*(_QWORD *)(***(_QWORD ***)v11 + 16LL) + 40LL) )
          v10 = *(_QWORD *)(***(_QWORD ***)v11 + 16LL);
        else
          v10 = sub_157EBA0(v27);
      }
      if ( v26 )
      {
        v24 = v10;
        if ( !sub_15CCEE0(v9[25], v10, v26) )
          goto LABEL_10;
        v10 = v24;
      }
      v26 = v10;
LABEL_10:
      v3 = *a2;
      v7 = (unsigned int)++v8;
    }
    while ( v8 != (__int64)(a2[1] - *a2) >> 3 );
    a1 = v9;
LABEL_15:
    v12 = *(_BYTE **)(a3 + 8);
    if ( v12 == *(_BYTE **)(a3 + 16) )
    {
      sub_170B610(a3, v12, &v26);
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = v26;
        v12 = *(_BYTE **)(a3 + 8);
      }
      *(_QWORD *)(a3 + 8) = v12 + 8;
    }
    if ( v20 != (_DWORD)v23 )
    {
LABEL_20:
      result = ++v23;
      if ( v21 == v23 )
        return result;
LABEL_21:
      v3 = *a2;
      continue;
    }
    break;
  }
  v26 = 0;
  v13 = *a2;
  if ( a2[1] == *a2 )
    goto LABEL_34;
  v14 = 0;
  v15 = 0;
  v16 = a1;
  while ( 2 )
  {
    v18 = *(_QWORD *)(**(_QWORD **)(v13 + 8 * v14) + 8 * v23);
    sub_1C620D0(v16, *(unsigned int **)v18, *(__int64 ***)(v18 + 8), &v27, v28, v16[25], 0);
    v17 = v28[0];
    if ( *(_BYTE *)(v28[0] + 16) == 18 )
    {
      if ( v28[0] == *(_QWORD *)(*(_QWORD *)(***(_QWORD ***)(v18 + 8) + 16LL) + 40LL) )
        v17 = *(_QWORD *)(***(_QWORD ***)(v18 + 8) + 16LL);
      else
        v17 = sub_157EBA0(v28[0]);
    }
    if ( v26 )
    {
      v25 = v17;
      if ( sub_15CCEE0(v16[25], v17, v26) )
      {
        v17 = v25;
        goto LABEL_28;
      }
    }
    else
    {
LABEL_28:
      v26 = v17;
    }
    v13 = *a2;
    v14 = (unsigned int)++v15;
    if ( v15 != (__int64)(a2[1] - *a2) >> 3 )
      continue;
    break;
  }
  a1 = v16;
LABEL_34:
  v19 = *(_BYTE **)(a3 + 8);
  if ( v19 == *(_BYTE **)(a3 + 16) )
  {
    sub_170B610(a3, v19, &v26);
    goto LABEL_20;
  }
  if ( v19 )
  {
    *(_QWORD *)v19 = v26;
    v19 = *(_BYTE **)(a3 + 8);
  }
  ++v23;
  *(_QWORD *)(a3 + 8) = v19 + 8;
  result = v23;
  if ( v21 != v23 )
    goto LABEL_21;
  return result;
}
