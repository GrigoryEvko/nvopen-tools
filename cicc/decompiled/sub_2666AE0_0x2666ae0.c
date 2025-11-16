// Function: sub_2666AE0
// Address: 0x2666ae0
//
unsigned __int64 __fastcall sub_2666AE0(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  __int64 *v5; // r12
  __int64 v6; // rbx
  char v7; // al
  __int64 v8; // r14
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 (__fastcall *v12)(__int64, _BYTE *, __int64, __int64); // rax
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdi
  _BYTE *v16; // rbx
  __int64 (__fastcall *v17)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // r12
  __int64 v24; // rbx
  __int64 v25; // rdx
  unsigned int v26; // esi
  _QWORD *v27; // rax
  __int64 v28; // r15
  __int64 v29; // r14
  __int64 v30; // rbx
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 *v33; // [rsp+0h] [rbp-D0h]
  __int64 v34; // [rsp+10h] [rbp-C0h]
  unsigned int v35; // [rsp+1Ch] [rbp-B4h]
  __int64 v36; // [rsp+20h] [rbp-B0h]
  unsigned int v37; // [rsp+3Ch] [rbp-94h] BYREF
  int v38[8]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v39; // [rsp+60h] [rbp-70h]
  _BYTE v40[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v41; // [rsp+90h] [rbp-40h]

  v5 = a1;
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_BYTE *)(v6 + 8);
  if ( v7 == 15 )
  {
    v37 = 0;
    v8 = sub_ACADE0((__int64 **)a3);
    v35 = *(_DWORD *)(v6 + 12);
    if ( !v35 )
      return v8;
    v34 = a3;
    v10 = 0;
    while ( 1 )
    {
      v11 = v5[10];
      v36 = *(_QWORD *)(*(_QWORD *)(v34 + 16) + 8 * v10);
      v39 = 257;
      v12 = *(__int64 (__fastcall **)(__int64, _BYTE *, __int64, __int64))(*(_QWORD *)v11 + 80LL);
      if ( v12 != sub_92FAE0 )
        break;
      if ( *(_BYTE *)a2 <= 0x15u )
      {
        v13 = sub_AAADB0(a2, &v37, 1);
        goto LABEL_15;
      }
LABEL_24:
      v41 = 257;
      v13 = (__int64)sub_BD2C40(104, unk_3F10A14);
      if ( v13 )
      {
        v19 = sub_B501B0(*(_QWORD *)(a2 + 8), &v37, 1);
        sub_B44260(v13, v19, 64, 1u, 0, 0);
        if ( *(_QWORD *)(v13 - 32) )
        {
          v20 = *(_QWORD *)(v13 - 24);
          **(_QWORD **)(v13 - 16) = v20;
          if ( v20 )
            *(_QWORD *)(v20 + 16) = *(_QWORD *)(v13 - 16);
        }
        *(_QWORD *)(v13 - 32) = a2;
        v21 = *(_QWORD *)(a2 + 16);
        *(_QWORD *)(v13 - 24) = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = v13 - 24;
        *(_QWORD *)(v13 - 16) = a2 + 16;
        *(_QWORD *)(a2 + 16) = v13 - 32;
        *(_QWORD *)(v13 + 72) = v13 + 88;
        *(_QWORD *)(v13 + 80) = 0x400000000LL;
        sub_B50030(v13, &v37, 1, (__int64)v40);
      }
      (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v5[11] + 16LL))(
        v5[11],
        v13,
        v38,
        v5[7],
        v5[8]);
      v22 = *v5 + 16LL * *((unsigned int *)v5 + 2);
      if ( *v5 != v22 )
      {
        v33 = v5;
        v23 = *v5;
        v24 = v22;
        do
        {
          v25 = *(_QWORD *)(v23 + 8);
          v26 = *(_DWORD *)v23;
          v23 += 16;
          sub_B99FD0(v13, v26, v25);
        }
        while ( v24 != v23 );
        v5 = v33;
      }
LABEL_16:
      v14 = sub_2666AE0(v5, v13, v36);
      v15 = v5[10];
      v16 = (_BYTE *)v14;
      v39 = 257;
      v17 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v15 + 88LL);
      if ( v17 == sub_9482E0 )
      {
        if ( *(_BYTE *)v8 > 0x15u || *v16 > 0x15u )
        {
LABEL_35:
          v41 = 257;
          v27 = sub_BD2C40(104, unk_3F148BC);
          v28 = (__int64)v27;
          if ( v27 )
          {
            sub_B44260((__int64)v27, *(_QWORD *)(v8 + 8), 65, 2u, 0, 0);
            *(_QWORD *)(v28 + 72) = v28 + 88;
            *(_QWORD *)(v28 + 80) = 0x400000000LL;
            sub_B4FD20(v28, v8, (__int64)v16, &v37, 1, (__int64)v40);
          }
          (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v5[11] + 16LL))(
            v5[11],
            v28,
            v38,
            v5[7],
            v5[8]);
          v29 = *v5;
          v30 = *v5 + 16LL * *((unsigned int *)v5 + 2);
          if ( *v5 != v30 )
          {
            do
            {
              v31 = *(_QWORD *)(v29 + 8);
              v32 = *(_DWORD *)v29;
              v29 += 16;
              sub_B99FD0(v28, v32, v31);
            }
            while ( v30 != v29 );
          }
          v8 = v28;
          goto LABEL_22;
        }
        v18 = sub_AAAE30(v8, (__int64)v16, &v37, 1);
      }
      else
      {
        v18 = v17(v15, (_BYTE *)v8, v16, (__int64)&v37, 1);
      }
      if ( !v18 )
        goto LABEL_35;
      v8 = v18;
LABEL_22:
      v10 = v37 + 1;
      v37 = v10;
      if ( (unsigned int)v10 >= v35 )
        return v8;
    }
    v13 = v12(v11, (_BYTE *)a2, (__int64)&v37, 1);
LABEL_15:
    if ( v13 )
      goto LABEL_16;
    goto LABEL_24;
  }
  if ( v7 == 12 )
  {
    if ( *(_BYTE *)(a3 + 8) == 14 )
    {
      v41 = 257;
      return sub_2666940(a1, 0x30u, a2, (__int64 **)a3, (__int64)v40, 0, v38[0], 0);
    }
  }
  else if ( v7 == 14 && *(_BYTE *)(a3 + 8) == 12 )
  {
    v41 = 257;
    return sub_2666940(a1, 0x2Fu, a2, (__int64 **)a3, (__int64)v40, 0, v38[0], 0);
  }
  v41 = 257;
  return sub_2666940(a1, 0x31u, a2, (__int64 **)a3, (__int64)v40, 0, v38[0], 0);
}
