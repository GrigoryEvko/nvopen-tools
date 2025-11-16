// Function: sub_7BE320
// Address: 0x7be320
//
_QWORD *__fastcall sub_7BE320(int a1, unsigned int *a2)
{
  _BYTE *v2; // rbx
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // r12
  unsigned __int8 v19; // r15
  _QWORD *v20; // r12
  _QWORD **v21; // r14
  _BYTE *v22; // rbx
  __int64 v23; // rsi
  _QWORD *v25; // rbx
  _QWORD *v26; // rax
  _BYTE *v27; // r14
  _QWORD *v28; // rax
  __int64 v29; // [rsp+8h] [rbp-58h]
  int i; // [rsp+1Ch] [rbp-44h] BYREF
  _QWORD *v31; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v32[7]; // [rsp+28h] [rbp-38h] BYREF

  v31 = 0;
  v2 = sub_724D80(0);
  sub_7B8B50(0, a2, v3, v4, v5, v6);
  v11 = qword_4F061C8;
  ++*(_BYTE *)(qword_4F061C8 + 84LL);
  ++*(_BYTE *)(v11 + 36);
  if ( word_4F06418[0] == 27 )
  {
    sub_7B8B50(0, a2, v7, v8, v9, v10);
  }
  else
  {
    a2 = &dword_4F063F8;
    sub_6851C0(0x7Du, &dword_4F063F8);
  }
  v32[0] = *(_QWORD *)&dword_4F063F8;
  sub_6BA680((__int64)v2);
  sub_6959C0((__int64)v2, (__int64)a2);
  sub_7BE280(0x1Cu, 18, 0, 0, v12, v13);
  sub_7BE280(0x4Cu, 2294, 0, 0, v14, v15);
  v16 = qword_4F061C8;
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_BYTE *)(v16 + 84);
  v17 = v2[173];
  if ( !v17 )
    goto LABEL_17;
  if ( v17 == 12 )
  {
    v27 = sub_724D80(12);
    *((_QWORD *)v27 + 16) = *((_QWORD *)v2 + 16);
    sub_7249B0((__int64)v27, 12);
    *((_QWORD *)v27 + 23) = v2;
    v28 = sub_725090(1u);
    *((_BYTE *)v28 + 24) |= 0x40u;
    v20 = v28;
    v31 = v28;
    v28[4] = v27;
LABEL_18:
    if ( !a1 )
      return v20;
    goto LABEL_19;
  }
  if ( v17 != 1 )
    goto LABEL_16;
  v18 = *((_QWORD *)v2 + 16);
  for ( i = 0; *(_BYTE *)(v18 + 140) == 12; v18 = *(_QWORD *)(v18 + 160) )
    ;
  v29 = sub_620FA0((__int64)v2, &i);
  if ( *(_BYTE *)(v18 + 140) != 2 )
  {
LABEL_16:
    sub_6851C0(0x295u, v32);
LABEL_17:
    v31 = sub_725090(1u);
    v20 = v31;
    v31[4] = v2;
    goto LABEL_18;
  }
  if ( i )
  {
    sub_6851C0(0x3Du, v32);
    goto LABEL_17;
  }
  if ( v29 < 0 )
  {
    sub_6851C0(0xB3Bu, v32);
    goto LABEL_17;
  }
  v19 = *(_BYTE *)(v18 + 160);
  v20 = 0;
  v21 = &v31;
  if ( v29 )
  {
    do
    {
      v22 = sub_724D80(1);
      v23 = (__int64)v20;
      *v21 = sub_725090(1u);
      v20 = (_QWORD *)((char *)v20 + 1);
      sub_72BAF0((__int64)v22, v23, v19);
      (*v21)[4] = v22;
      v21 = (_QWORD **)*v21;
    }
    while ( v29 > (__int64)v20 );
    v20 = v31;
    if ( a1 )
    {
      if ( v31 )
      {
LABEL_19:
        v25 = v20;
        do
        {
          v26 = sub_6E6B30(v25[4]);
          v25[4] = 0;
          v25[6] = v26;
          v25 = (_QWORD *)*v25;
        }
        while ( v25 );
      }
    }
  }
  return v20;
}
