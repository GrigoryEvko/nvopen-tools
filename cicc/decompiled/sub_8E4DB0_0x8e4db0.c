// Function: sub_8E4DB0
// Address: 0x8e4db0
//
__int64 __fastcall sub_8E4DB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v5; // rcx
  __int64 v6; // r8
  _QWORD *v7; // rbx
  _QWORD *v8; // r13
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-40h]

  if ( *(char *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 181LL) >= 0
    || *(char *)(*(_QWORD *)(*(_QWORD *)a2 + 96LL) + 181LL) >= 0
    || sub_8D3B10(a1)
    || sub_8D3B10(a2) )
  {
    return 0;
  }
  v7 = **(_QWORD ***)(a1 + 168);
  v8 = **(_QWORD ***)(a2 + 168);
  if ( v7 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v7[5] + 160LL);
      if ( v9 )
        break;
      v7 = (_QWORD *)*v7;
      if ( !v7 )
        goto LABEL_10;
    }
  }
  else
  {
LABEL_10:
    v9 = *(_QWORD *)(a1 + 160);
    v7 = 0;
  }
  if ( v8 )
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v8[5] + 160LL);
      if ( v10 )
        break;
      v8 = (_QWORD *)*v8;
      if ( !v8 )
        goto LABEL_14;
    }
  }
  else
  {
LABEL_14:
    v8 = 0;
    v10 = *(_QWORD *)(a2 + 160);
    if ( !v10 )
      return 0;
  }
  if ( v9 )
  {
    v2 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = v10;
        if ( !sub_8E4D50(v9, v10, v10, v5, v6) || ((*(_BYTE *)(v13 + 146) ^ *(_BYTE *)(v9 + 146)) & 4) != 0 )
          return v2;
        v2 = *(_QWORD *)(v9 + 128) + 1LL;
        v9 = sub_72FD90(*(_QWORD *)(v9 + 112), 6);
        v11 = *(_QWORD *)(v13 + 112);
        if ( !v9 )
          break;
LABEL_23:
        v10 = sub_72FD90(v11, 6);
        if ( !v10 )
          goto LABEL_26;
      }
      if ( !v7 )
        break;
      while ( 1 )
      {
        v7 = (_QWORD *)*v7;
        if ( !v7 )
          break;
        v9 = *(_QWORD *)(v7[5] + 160LL);
        if ( v9 )
          goto LABEL_23;
      }
      v9 = *(_QWORD *)(a1 + 160);
      v12 = sub_72FD90(v11, 6);
      v7 = (_QWORD *)v12;
      if ( !v12 )
        goto LABEL_26;
      v10 = v12;
      v7 = 0;
LABEL_31:
      if ( !v9 || !v10 )
        return v2;
    }
    v9 = sub_72FD90(v11, 6);
    if ( v9 )
      return v2;
    v7 = 0;
LABEL_26:
    if ( !v8 )
      return v2;
    v8 = (_QWORD *)*v8;
    if ( v8 )
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v8[5] + 160LL);
        if ( v10 )
          break;
        v8 = (_QWORD *)*v8;
        if ( !v8 )
          goto LABEL_30;
      }
    }
    else
    {
LABEL_30:
      v10 = *(_QWORD *)(a2 + 160);
    }
    goto LABEL_31;
  }
  return 0;
}
