// Function: sub_24E58A0
// Address: 0x24e58a0
//
void __fastcall sub_24E58A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // rax
  char v8; // al
  _QWORD *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rbx
  __int64 v12; // r12
  unsigned int *v13; // r15
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r12
  unsigned int v19; // r8d
  __int64 v20; // rax
  char v22; // [rsp+18h] [rbp-70h]
  _QWORD v23[4]; // [rsp+28h] [rbp-60h] BYREF
  __int16 v24; // [rsp+48h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 288);
  v23[0] = "ResumeFn.addr";
  v24 = 259;
  v5 = sub_9213A0((unsigned int **)a1, v4, a3, 0, 0, (__int64)v23, 7u);
  v6 = sub_AC9EC0(**(__int64 ****)(*(_QWORD *)(a2 + 288) + 16LL));
  v7 = sub_AA4E30(*(_QWORD *)(a1 + 48));
  v8 = sub_AE5020(v7, *(_QWORD *)(v6 + 8));
  v24 = 257;
  v22 = v8;
  v9 = sub_BD2C40(80, unk_3F10A10);
  v11 = (__int64)v9;
  if ( v9 )
    sub_B4D3C0((__int64)v9, v6, v5, 0, v22, v10, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 88) + 16LL))(
    *(_QWORD *)(a1 + 88),
    v11,
    v23,
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 64));
  v12 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v12 )
  {
    v13 = *(unsigned int **)a1;
    do
    {
      v14 = *((_QWORD *)v13 + 1);
      v15 = *v13;
      v13 += 4;
      sub_B99FD0(v11, v15, v14);
    }
    while ( (unsigned int *)v12 != v13 );
  }
  if ( *(_BYTE *)(a2 + 365) )
  {
    if ( *(_BYTE *)(a2 + 364) )
    {
      v16 = sub_ACD640(
              *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 288) + 16LL) + 8LL * *(unsigned int *)(a2 + 352)),
              *(unsigned int *)(a2 + 128) - 1LL,
              0);
      v17 = *(_QWORD *)(a2 + 288);
      v18 = v16;
      v19 = *(_DWORD *)(a2 + 352);
      v23[0] = "index.addr";
      v24 = 259;
      v20 = sub_9213A0((unsigned int **)a1, v17, a3, 0, v19, (__int64)v23, 7u);
      sub_2463EC0((__int64 *)a1, v18, v20, 0, 0);
    }
  }
}
