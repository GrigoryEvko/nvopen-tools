// Function: sub_2C135E0
// Address: 0x2c135e0
//
__int64 __fastcall sub_2C135E0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // r12
  __int64 v6; // rsi
  __int64 v7; // r10
  _BYTE *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v15; // [rsp+8h] [rbp-B8h]
  __int64 v16; // [rsp+10h] [rbp-B0h]
  __int64 v17; // [rsp+18h] [rbp-A8h]
  unsigned int v18; // [rsp+20h] [rbp-A0h]
  __int64 v19; // [rsp+28h] [rbp-98h]
  _QWORD v20[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v21[4]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v22[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v23; // [rsp+80h] [rbp-40h]

  v22[0] = *(_QWORD *)(a1 + 88);
  if ( v22[0] )
    sub_2AAAFA0(v22);
  sub_2BF1A90(a2, (__int64)v22);
  sub_9C6650(v22);
  v17 = *(_QWORD *)(a2 + 904);
  v16 = sub_2BFB640(a2, **(_QWORD **)(a1 + 48), 0);
  v3 = sub_2BFB640(a2, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL), 1);
  v4 = *(_QWORD *)(v16 + 8);
  v5 = v3;
  if ( *(_DWORD *)(a1 + 56) != 3 || (v6 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL)) == 0 )
  {
    v23 = 257;
    v10 = sub_BCB2A0(*(_QWORD **)(v17 + 72));
    v11 = sub_ACD640(v10, 1, 0);
    BYTE4(v19) = *(_BYTE *)(v4 + 8) == 18;
    LODWORD(v19) = *(_DWORD *)(v4 + 32);
    v7 = sub_B37620((unsigned int **)v17, v19, v11, v22);
    if ( *(_DWORD *)(a1 + 96) != 15 )
      goto LABEL_8;
    goto LABEL_6;
  }
  v7 = sub_2BFB640(a2, v6, 0);
  if ( *(_DWORD *)(a1 + 96) == 15 )
  {
LABEL_6:
    v15 = v7;
    v23 = 257;
    v8 = (_BYTE *)sub_AD6530(*(_QWORD *)(v5 + 8), 257);
    v9 = sub_929DE0((unsigned int **)v17, v8, (_BYTE *)v5, (__int64)v22, 0, 0);
    v7 = v15;
    v5 = v9;
  }
LABEL_8:
  v21[1] = v5;
  v12 = *(_QWORD *)(a2 + 904);
  v23 = 257;
  v21[2] = v7;
  v21[0] = v16;
  v13 = *(_QWORD *)(v5 + 8);
  v20[0] = v4;
  v20[1] = v13;
  return sub_B33D10(v12, 0xA1u, (__int64)v20, 2, (int)v21, 3, v18, (__int64)v22);
}
