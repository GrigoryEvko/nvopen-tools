// Function: sub_1100C60
// Address: 0x1100c60
//
unsigned __int8 *__fastcall sub_1100C60(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // r13d
  int v5; // ebx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 *v9; // r9
  __int64 v10; // r15
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 *v13; // r13
  int v14; // edx
  __int64 *v15; // r15
  __int64 v16; // r14
  __int64 v17; // rbx
  unsigned int v18; // eax
  __int64 v19; // r13
  _QWORD *v20; // rax
  _QWORD *v21; // r12
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r13
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // r13
  __int64 v29; // r14
  __int64 v30; // rdx
  unsigned int v31; // esi
  unsigned int v32; // [rsp+8h] [rbp-A8h]
  _QWORD *v33; // [rsp+8h] [rbp-A8h]
  __int64 v34; // [rsp+18h] [rbp-98h]
  _BYTE v35[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v36; // [rsp+40h] [rbp-70h]
  _BYTE v37[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v38; // [rsp+70h] [rbp-40h]

  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  v4 = *(_DWORD *)(v3 + 8) >> 8;
  v5 = sub_BCB060(*(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL));
  if ( v5 == sub_AE2980(a1[11], v4)[1] )
    return sub_11005E0(a1, (unsigned __int8 *)a2, v6, v7, v8, v9);
  v10 = a1[11];
  v11 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
  v12 = sub_BD5C60(a2);
  v13 = (__int64 *)sub_AE4420(v10, v12, v4);
  v14 = *(unsigned __int8 *)(v11 + 8);
  if ( (unsigned int)(v14 - 17) <= 1 )
  {
    BYTE4(v34) = (_BYTE)v14 == 18;
    LODWORD(v34) = *(_DWORD *)(v11 + 32);
    v13 = (__int64 *)sub_BCE1B0(v13, v34);
  }
  v15 = (__int64 *)a1[4];
  v16 = *(_QWORD *)(a2 - 32);
  v36 = 257;
  v17 = *(_QWORD *)(v16 + 8);
  v32 = sub_BCB060(v17);
  v18 = sub_BCB060((__int64)v13);
  if ( v32 < v18 )
  {
    if ( v13 == (__int64 *)v17 )
      goto LABEL_9;
    v23 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *))(*(_QWORD *)v15[10] + 120LL))(
            v15[10],
            39,
            v16,
            v13);
    if ( !v23 )
    {
      v38 = 257;
      v33 = sub_BD2C40(72, unk_3F10A14);
      if ( v33 )
        sub_B515B0((__int64)v33, v16, (__int64)v13, (__int64)v37, 0, 0);
      (*(void (__fastcall **)(__int64, _QWORD *, _BYTE *, __int64, __int64))(*(_QWORD *)v15[11] + 16LL))(
        v15[11],
        v33,
        v35,
        v15[7],
        v15[8]);
      v28 = *v15;
      v29 = *v15 + 16LL * *((unsigned int *)v15 + 2);
      if ( *v15 != v29 )
      {
        do
        {
          v30 = *(_QWORD *)(v28 + 8);
          v31 = *(_DWORD *)v28;
          v28 += 16;
          sub_B99FD0((__int64)v33, v31, v30);
        }
        while ( v29 != v28 );
      }
      v16 = (__int64)v33;
      goto LABEL_9;
    }
LABEL_14:
    v16 = v23;
    goto LABEL_9;
  }
  if ( v32 != v18 && v13 != (__int64 *)v17 )
  {
    v23 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64 *))(*(_QWORD *)v15[10] + 120LL))(
            v15[10],
            38,
            v16,
            v13);
    if ( !v23 )
    {
      v38 = 257;
      v16 = sub_B51D30(38, v16, (__int64)v13, (__int64)v37, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v15[11] + 16LL))(
        v15[11],
        v16,
        v35,
        v15[7],
        v15[8]);
      v24 = *v15;
      v25 = *v15 + 16LL * *((unsigned int *)v15 + 2);
      if ( *v15 != v25 )
      {
        do
        {
          v26 = *(_QWORD *)(v24 + 8);
          v27 = *(_DWORD *)v24;
          v24 += 16;
          sub_B99FD0(v16, v27, v26);
        }
        while ( v25 != v24 );
      }
      goto LABEL_9;
    }
    goto LABEL_14;
  }
LABEL_9:
  v19 = *(_QWORD *)(a2 + 8);
  v38 = 257;
  v20 = sub_BD2C40(72, unk_3F10A14);
  v21 = v20;
  if ( v20 )
    sub_B51B50((__int64)v20, v16, v19, (__int64)v37, 0, 0);
  return (unsigned __int8 *)v21;
}
