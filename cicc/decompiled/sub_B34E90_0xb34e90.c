// Function: sub_B34E90
// Address: 0xb34e90
//
__int64 __fastcall sub_B34E90(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v12; // rax
  char v14; // al
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  char v19; // [rsp+7h] [rbp-99h]
  __int64 v20; // [rsp+8h] [rbp-98h]
  _QWORD v21[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v22[4]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v23; // [rsp+40h] [rbp-60h] BYREF
  __int16 v24; // [rsp+60h] [rbp-40h]

  v7 = a5;
  v8 = *(_QWORD *)(a3 + 8);
  v9 = *(_QWORD *)(a2 + 8);
  if ( !a5 )
  {
    v14 = *(_BYTE *)(v8 + 8);
    v19 = a4;
    v15 = *(_QWORD *)(a1 + 72);
    v20 = *(_QWORD *)(a2 + 8);
    LODWORD(v23) = *(_DWORD *)(v8 + 32);
    BYTE4(v23) = v14 == 18;
    v16 = sub_BCB2A0(v15);
    v17 = sub_BCE1B0(v16, v23);
    v18 = sub_AD62B0(v17);
    a4 = v19;
    v9 = v20;
    v7 = v18;
  }
  v22[1] = a3;
  v10 = *(_QWORD *)(a1 + 72);
  v21[0] = v9;
  v21[1] = v8;
  v11 = (unsigned int)(1LL << a4);
  v22[0] = a2;
  v12 = sub_BCB2D0(v10);
  v22[2] = sub_ACD640(v12, v11, 0);
  v24 = 257;
  v22[3] = v7;
  return sub_B34BE0(a1, 0xE5u, (int)v22, 4, (__int64)v21, 2, (__int64)&v23);
}
