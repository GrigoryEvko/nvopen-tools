// Function: sub_28EE010
// Address: 0x28ee010
//
__int64 __fastcall sub_28EE010(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v14; // rbx
  _QWORD *v15; // r12
  __int64 v16; // rax
  _QWORD *v17; // rbx
  _QWORD *v18; // r12
  __int64 v19; // rax
  __int64 v20; // [rsp+0h] [rbp-70h] BYREF
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 v22; // [rsp+10h] [rbp-60h]
  __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+28h] [rbp-48h]
  __int64 v25; // [rsp+30h] [rbp-40h]

  v2 = a1 + 320;
  v3 = a1 + 896;
  *(_QWORD *)a1 = off_4A21D00;
  do
  {
    v4 = *(unsigned int *)(v3 + 24);
    if ( (_DWORD)v4 )
    {
      v5 = *(_QWORD **)(v3 + 8);
      v6 = &v5[9 * v4];
      do
      {
        while ( *v5 == -4096 )
        {
          if ( v5[1] != -4096 )
            goto LABEL_5;
          v5 += 9;
          if ( v6 == v5 )
            goto LABEL_15;
        }
        if ( *v5 != -8192 || v5[1] != -8192 )
        {
LABEL_5:
          v7 = v5[7];
          if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
            sub_BD60C0(v5 + 5);
          v8 = v5[4];
          if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
            sub_BD60C0(v5 + 2);
        }
        v5 += 9;
      }
      while ( v6 != v5 );
    }
LABEL_15:
    v9 = *(unsigned int *)(v3 + 24);
    v10 = *(_QWORD *)(v3 + 8);
    v3 -= 32;
    sub_C7D6A0(v10, 72 * v9, 8);
  }
  while ( v2 != v3 );
  sub_28EDAB0((unsigned __int64 *)(a1 + 272));
  v11 = *(unsigned int *)(a1 + 264);
  if ( (_DWORD)v11 )
  {
    v14 = *(_QWORD **)(a1 + 248);
    v20 = 0;
    v21 = 0;
    v22 = -4096;
    v15 = &v14[3 * v11];
    v23 = 0;
    v24 = 0;
    v25 = -8192;
    do
    {
      v16 = v14[2];
      if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
        sub_BD60C0(v14);
      v14 += 3;
    }
    while ( v15 != v14 );
    if ( v22 != -4096 && v22 != 0 )
      sub_BD60C0(&v20);
    v11 = *(unsigned int *)(a1 + 264);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 248), 24 * v11, 8);
  v12 = *(unsigned int *)(a1 + 232);
  if ( (_DWORD)v12 )
  {
    v17 = *(_QWORD **)(a1 + 216);
    v20 = 0;
    v21 = 0;
    v22 = -4096;
    v18 = &v17[4 * v12];
    v23 = 0;
    v24 = 0;
    v25 = -8192;
    do
    {
      v19 = v17[2];
      if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
        sub_BD60C0(v17);
      v17 += 4;
    }
    while ( v18 != v17 );
    if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
      sub_BD60C0(&v23);
    if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
      sub_BD60C0(&v20);
    LODWORD(v12) = *(_DWORD *)(a1 + 232);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 216), 32LL * (unsigned int)v12, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 184), 16LL * *(unsigned int *)(a1 + 200), 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
