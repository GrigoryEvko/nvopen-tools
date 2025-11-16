// Function: sub_836920
// Address: 0x836920
//
__int64 __fastcall sub_836920(__int64 a1, __int64 a2, int a3, unsigned int a4, __int64 a5, _DWORD *a6, __int64 a7)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 *v13; // r9
  __int64 v14; // r15
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // r13
  char v18; // r12
  __int64 v19; // r15
  __int64 v20; // rcx
  __int64 v21; // rbx
  __int64 v22; // r8
  __int64 *v23; // r9
  __int64 v24; // rdx
  __int64 v25; // r12
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 *v29; // r9
  __int64 v31; // rdi
  __int64 v32; // r8
  __int64 v33; // r9
  _QWORD *v34; // r12
  _QWORD *v35; // rbx
  __int64 v38; // [rsp+8h] [rbp-88h]
  int v41; // [rsp+2Ch] [rbp-64h] BYREF
  int v42; // [rsp+30h] [rbp-60h] BYREF
  int v43; // [rsp+34h] [rbp-5Ch] BYREF
  _QWORD *v44; // [rsp+38h] [rbp-58h] BYREF
  _BYTE v45[80]; // [rsp+40h] [rbp-50h] BYREF

  v8 = a1;
  v9 = sub_82BD70();
  v14 = *(_QWORD *)(v9 + 1024);
  v15 = v9;
  if ( v14 == *(_QWORD *)(v9 + 1016) )
    sub_8332F0(v9, a2, v10, v11, v12, v13);
  v16 = *(_QWORD *)(v15 + 1008) + 40 * v14;
  if ( v16 )
  {
    *(_BYTE *)v16 &= 0xFCu;
    *(_QWORD *)(v16 + 8) = 0;
    *(_QWORD *)(v16 + 16) = 0;
    *(_QWORD *)(v16 + 24) = 0;
    *(_QWORD *)(v16 + 32) = 0;
  }
  *(_QWORD *)(v15 + 1024) = v14 + 1;
  *a6 = 0;
  if ( *(_BYTE *)(a1 + 140) == 12 )
  {
    do
      v8 = *(_QWORD *)(v8 + 160);
    while ( *(_BYTE *)(v8 + 140) == 12 );
  }
  if ( (unsigned int)sub_8D23B0(v8) )
  {
    if ( (unsigned int)sub_8D3A70(v8) )
      sub_8AD220(v8, 0);
  }
  while ( *(_BYTE *)(v8 + 140) == 12 )
    v8 = *(_QWORD *)(v8 + 160);
  v38 = *(_QWORD *)(*(_QWORD *)v8 + 96LL);
  v17 = sub_82C1B0(*(_QWORD *)(v38 + 8), 0, 0, (__int64)v45);
  if ( !v17 )
  {
    v19 = 0;
    goto LABEL_24;
  }
  v18 = 0;
  v19 = 0;
  do
  {
    while ( 1 )
    {
      if ( *(_BYTE *)(v17 + 80) == 20 )
      {
        if ( (_DWORD)a2 )
          v18 = 1;
        goto LABEL_15;
      }
      if ( (unsigned int)sub_72F310(*(_QWORD *)(v17 + 88), a3) && (*(char *)(*(_QWORD *)(v17 + 88) + 193LL) >= 0 || !a4) )
        break;
LABEL_15:
      v17 = sub_82C230(v45);
      if ( !v17 )
        goto LABEL_22;
    }
    if ( v19 )
      goto LABEL_27;
    v19 = v17;
    v17 = sub_82C230(v45);
  }
  while ( v17 );
LABEL_22:
  if ( v19 || (v18 & 1) == 0 )
    goto LABEL_24;
LABEL_27:
  v41 = 0;
  v31 = *(_QWORD *)(v38 + 8);
  v42 = 0;
  v44 = 0;
  sub_8360D0(v31, 0, 0, 0, 0, 0, 0, 0, a4, 0, 0, 0, a2 == 0, 0, 0, 0, 0, (__int64 *)&v44, a7, &v41, &v42);
  sub_82D8D0((__int64 *)&v44, a5, &v43, a6, v32, v33);
  if ( v43 )
  {
    v34 = v44;
    v19 = 0;
    if ( !v44 )
      goto LABEL_24;
  }
  else
  {
    v34 = v44;
    v19 = 0;
    if ( !v44 )
      goto LABEL_24;
    if ( !*a6 )
      v19 = v44[1];
  }
  do
  {
    v35 = v34;
    v34 = (_QWORD *)*v34;
    sub_725130((__int64 *)v35[5]);
    sub_82D8A0((_QWORD *)v35[15]);
    *v35 = qword_4D03C68;
    qword_4D03C68 = v35;
  }
  while ( v34 );
LABEL_24:
  v21 = sub_82BD70();
  v24 = *(_QWORD *)(v21 + 1008);
  v25 = *(_QWORD *)(v24 + 8 * (5LL * *(_QWORD *)(v21 + 1024) - 5) + 32);
  if ( v25 )
  {
    sub_823A00(*(_QWORD *)v25, 16LL * (unsigned int)(*(_DWORD *)(v25 + 8) + 1), v24, v20, v22, v23);
    sub_823A00(v25, 16, v26, v27, v28, v29);
  }
  --*(_QWORD *)(v21 + 1024);
  return v19;
}
