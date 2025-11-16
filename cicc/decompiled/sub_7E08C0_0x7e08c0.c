// Function: sub_7E08C0
// Address: 0x7e08c0
//
void __fastcall sub_7E08C0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 **v9; // rax
  __int64 *v10; // rdx
  _QWORD *v11; // rax
  _QWORD *v12; // rdx

  v3 = *(_QWORD **)(a1 + 168);
  if ( !a2 )
  {
    if ( v3[3] )
    {
      ((void (*)(void))sub_7E08C0)();
      v8 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 16LL);
      if ( v8 )
        goto LABEL_6;
    }
    else
    {
      v8 = v3[2];
      if ( v8 )
        goto LABEL_6;
    }
    goto LABEL_13;
  }
  v4 = *(_QWORD *)(a2 + 40);
  v5 = *(_QWORD *)(v4 + 168);
  v6 = *(_QWORD *)(v5 + 24);
  if ( !v6 )
  {
    v8 = *(_QWORD *)(v5 + 16);
    if ( v8 )
      goto LABEL_6;
    goto LABEL_18;
  }
  v7 = sub_8E5650(v6);
  sub_7E08C0(a1, v7);
  v8 = *(_QWORD *)(*(_QWORD *)(v4 + 168) + 16LL);
  if ( !v8 )
    goto LABEL_18;
  do
  {
LABEL_6:
    while ( 1 )
    {
      if ( (*(_BYTE *)(v8 + 96) & 2) != 0 )
      {
        v9 = *(__int64 ***)(a1 + 168);
        do
        {
          do
            v9 = (__int64 **)*v9;
          while ( *(__int64 **)(v8 + 40) != v9[5] );
        }
        while ( ((_BYTE)v9[12] & 2) == 0 );
        if ( !v9[18] )
          break;
      }
      v8 = *(_QWORD *)(v8 + 16);
      if ( !v8 )
        goto LABEL_12;
    }
    v10 = (__int64 *)v3[6];
    v3[6] = (char *)v10 - 1;
    v9[18] = v10;
    v8 = *(_QWORD *)(v8 + 16);
  }
  while ( v8 );
LABEL_12:
  if ( a2 )
  {
LABEL_18:
    if ( (*(_BYTE *)(a2 + 96) & 2) == 0 )
      return;
    goto LABEL_14;
  }
LABEL_13:
  v3[7] = v3[6];
LABEL_14:
  v11 = (_QWORD *)v3[8];
  if ( v11 )
  {
    do
    {
      v12 = v11;
      v11 = (_QWORD *)*v11;
    }
    while ( v11 );
  }
  else
  {
    v12 = v3 + 8;
  }
  sub_7E0660(a1, a2, v12, 1);
}
