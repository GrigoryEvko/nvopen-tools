// Function: sub_8CD200
// Address: 0x8cd200
//
__int64 __fastcall sub_8CD200(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rcx
  __int64 v4; // r12
  __int64 **v5; // rax
  __int64 *v6; // rsi
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // rdi
  __int64 *v10; // r14
  __int64 v11; // rbx
  unsigned __int8 v12; // dl
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 *v15; // r14
  __int64 *v16; // r15
  __int64 *v17; // rbx
  int v18; // r15d
  int v19; // eax
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdi
  int v24; // r15d
  unsigned int v25; // [rsp+Ch] [rbp-44h]
  _BOOL4 v26[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v2 = a1;
  v4 = (__int64)a1;
  v5 = (__int64 **)a1[4];
  if ( v5 )
    v2 = *v5;
  v6 = *(__int64 **)(a2 + 32);
  v7 = a2;
  if ( v6 )
    v7 = *v6;
  if ( v2 == (__int64 *)v7 )
    return 1;
  if ( v5 && (*v5 != a1 || v5[1] == a1) )
    return 0;
  v9 = *a1;
  v10 = *(__int64 **)a2;
  if ( *(_QWORD *)v4 )
  {
    if ( v10 )
    {
      v11 = sub_880F80(v9);
      if ( v11 == sub_880F80((__int64)v10) )
        return 0;
    }
  }
  if ( !sub_8C7520((__int64 **)v4, (__int64 **)a2) )
    return 0;
  v12 = *(_BYTE *)(a2 + 89);
  if ( ((v12 ^ *(_BYTE *)(v4 + 89)) & 4) != 0 || (*(_BYTE *)(v4 + 89) & 4) != 0 )
    return 0;
  v13 = *(_QWORD *)(v4 + 48);
  if ( v13 )
  {
    v14 = *(_QWORD *)(a2 + 48);
    if ( !v14 )
      return 0;
    if ( v13 == v14 )
    {
LABEL_23:
      v12 = *(_BYTE *)(a2 + 89);
      goto LABEL_24;
    }
    if ( !*qword_4D03FD0 || !(unsigned int)sub_8C7EB0(v13, v14, 0xBu) )
      return 0;
LABEL_22:
    if ( (*(_BYTE *)(v4 + 89) & 4) == 0 )
      goto LABEL_23;
    return 0;
  }
  v20 = *(_QWORD *)(v4 + 40);
  v21 = *(_QWORD *)(a2 + 40);
  if ( v20 && *(_BYTE *)(v20 + 28) == 3 )
  {
    v23 = *(_QWORD *)(v20 + 32);
    if ( !v21 || *(_BYTE *)(v21 + 28) != 3 )
    {
      if ( v23 )
        return 0;
      goto LABEL_24;
    }
    v22 = *(_QWORD *)(v21 + 32);
    if ( v23 )
    {
      if ( v23 == v22 )
        goto LABEL_23;
      if ( !*qword_4D03FD0 || !v22 || !(unsigned int)sub_8C7EB0(v23, v22, 0x1Cu) )
        return 0;
      goto LABEL_22;
    }
  }
  else
  {
    if ( !v21 || *(_BYTE *)(v21 + 28) != 3 )
      goto LABEL_24;
    v22 = *(_QWORD *)(v21 + 32);
  }
  if ( v22 )
    return 0;
LABEL_24:
  if ( (v12 & 4) != 0
    || *(_BYTE *)(v4 + 140) == 9 && (*(_BYTE *)(*(_QWORD *)(v4 + 168) + 109LL) & 0x20) != 0
    || *(_BYTE *)(a2 + 140) == 9 && (*(_BYTE *)(*(_QWORD *)(a2 + 168) + 109LL) & 0x20) != 0
    || qword_4F074B0 )
  {
    return 0;
  }
  v15 = *(__int64 **)(v4 + 32);
  if ( dword_4F077C4 == 2 )
  {
    v26[0] = v15 != 0;
    sub_8CA0A0(v4, 0);
    result = sub_8CD160(v4, (_QWORD *)a2, v26);
    if ( !(v26[0] | (unsigned int)result) && !qword_4F074B0 )
    {
      v25 = result;
      sub_8CA0A0(v4, 1u);
      sub_8CA0A0(v4, 0);
      return v25;
    }
  }
  else
  {
    v16 = *(__int64 **)(a2 + 32);
    if ( v15 )
    {
      if ( v16 )
      {
        if ( *(__int64 **)(*v15 + 32) != v15 )
        {
          sub_8CF610(v4);
          v15 = *(__int64 **)(v4 + 32);
        }
        v17 = *(__int64 **)(*v16 + 32);
        if ( v16 != v17 )
        {
          sub_8CF610(a2);
          v17 = *(__int64 **)(a2 + 32);
        }
        result = 1;
        if ( v15 != v17 )
        {
          v18 = sub_8C6530(6u, *v15);
          v19 = sub_8C6530(6u, *v17);
          if ( v18 > v19 || v18 == v19 && (v24 = sub_8C6530(6u, a2), v24 < (int)sub_8C6530(6u, v4)) )
          {
            v15 = *(__int64 **)(a2 + 32);
            v17 = *(__int64 **)(v4 + 32);
            v4 = a2;
          }
          result = sub_8CF610(v4);
          *v15 = *v17;
        }
      }
      else
      {
        return sub_8CD160(a2, (_QWORD *)v4, 0);
      }
    }
    else
    {
      return sub_8CD160(v4, (_QWORD *)a2, 0);
    }
  }
  return result;
}
