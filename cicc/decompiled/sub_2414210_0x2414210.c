// Function: sub_2414210
// Address: 0x2414210
//
void __fastcall sub_2414210(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  _QWORD *v4; // rax
  _DWORD *v5; // rax
  __int64 *v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int8 *v9; // r10
  unsigned __int8 *v10; // r12
  __int64 (__fastcall *v11)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rdi
  unsigned __int8 **v15; // rsi
  __int64 v16; // r12
  __int64 v17; // rdi
  _BYTE *v18; // rsi
  __int64 v19; // rdi
  _BYTE *v20; // rsi
  __int64 v21; // r12
  __int64 v22; // r13
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // rax
  __int64 v26; // [rsp+10h] [rbp-B0h] BYREF
  unsigned __int8 *v27[2]; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v28; // [rsp+28h] [rbp-98h] BYREF
  char v29[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v30; // [rsp+50h] [rbp-70h]
  _QWORD v31[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v32; // [rsp+80h] [rbp-40h]

  v4 = *(_QWORD **)a1;
  v27[0] = a2;
  v26 = a3;
  if ( *v4 > 4u )
  {
    v5 = *(_DWORD **)(a1 + 24);
    v6 = *(__int64 **)(a1 + 8);
    v30 = 257;
    v7 = sub_AD64C0(**(_QWORD **)(a1 + 16), *v5 >> 1, 0);
    v8 = v6[10];
    v9 = v27[0];
    v10 = (unsigned __int8 *)v7;
    v11 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v8 + 32LL);
    if ( v11 == sub_9201A0 )
    {
      if ( *v27[0] > 0x15u || *v10 > 0x15u )
        goto LABEL_24;
      if ( (unsigned __int8)sub_AC47B0(25) )
        v12 = sub_AD5570(25, (__int64)v27[0], v10, 0, 0);
      else
        v12 = sub_AABE40(0x19u, v27[0], v10);
      v9 = v27[0];
      v13 = v12;
    }
    else
    {
      v25 = v11(v8, 25u, v27[0], v10, 0, 0);
      v9 = v27[0];
      v13 = v25;
    }
    if ( v13 )
    {
LABEL_9:
      v14 = *(_QWORD *)(a1 + 32);
      v28 = v13;
      v15 = *(unsigned __int8 ***)(v14 + 8);
      if ( v15 == *(unsigned __int8 ***)(v14 + 16) )
      {
        sub_9281F0(v14, v15, v27);
      }
      else
      {
        if ( v15 )
        {
          *v15 = v27[0];
          v15 = *(unsigned __int8 ***)(v14 + 8);
        }
        *(_QWORD *)(v14 + 8) = v15 + 1;
      }
      v16 = *(_QWORD *)(a1 + 40);
      v31[0] = sub_2412A60(
                 **(_QWORD **)(a1 + 48),
                 **(_QWORD **)(a1 + 56),
                 *(_QWORD *)(*(_QWORD *)(a1 + 56) + 8LL),
                 **(_BYTE **)(a1 + 64),
                 *(__int64 **)(a1 + 72));
      sub_240DEA0(v16, v31);
      v17 = *(_QWORD *)(a1 + 32);
      v18 = *(_BYTE **)(v17 + 8);
      if ( v18 != *(_BYTE **)(v17 + 16) )
      {
        if ( v18 )
        {
          *(_QWORD *)v18 = v28;
          v18 = *(_BYTE **)(v17 + 8);
        }
LABEL_19:
        *(_QWORD *)(v17 + 8) = v18 + 8;
        goto LABEL_20;
      }
      sub_9281F0(v17, v18, &v28);
      goto LABEL_20;
    }
LABEL_24:
    v32 = 257;
    v13 = sub_B504D0(25, (__int64)v9, (__int64)v10, (__int64)v31, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v6[11] + 16LL))(
      v6[11],
      v13,
      v29,
      v6[7],
      v6[8]);
    v21 = *v6;
    v22 = *v6 + 16LL * *((unsigned int *)v6 + 2);
    while ( v22 != v21 )
    {
      v23 = *(_QWORD *)(v21 + 8);
      v24 = *(_DWORD *)v21;
      v21 += 16;
      sub_B99FD0(v13, v24, v23);
    }
    goto LABEL_9;
  }
  v17 = *(_QWORD *)(a1 + 32);
  v18 = *(_BYTE **)(v17 + 8);
  if ( v18 != *(_BYTE **)(v17 + 16) )
  {
    if ( v18 )
    {
      *(unsigned __int8 **)v18 = v27[0];
      v18 = *(_BYTE **)(v17 + 8);
    }
    goto LABEL_19;
  }
  sub_9281F0(v17, v18, v27);
LABEL_20:
  v19 = *(_QWORD *)(a1 + 40);
  v20 = *(_BYTE **)(v19 + 8);
  if ( v20 == *(_BYTE **)(v19 + 16) )
  {
    sub_9281F0(v19, v20, &v26);
  }
  else
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = v26;
      v20 = *(_BYTE **)(v19 + 8);
    }
    *(_QWORD *)(v19 + 8) = v20 + 8;
  }
}
