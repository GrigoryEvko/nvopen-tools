// Function: sub_37C70E0
// Address: 0x37c70e0
//
__int64 __fastcall sub_37C70E0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // rdx
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  __int128 v15; // [rsp-20h] [rbp-60h]
  _DWORD v16[3]; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v17; // [rsp+10h] [rbp-30h]

  v2 = *(_QWORD *)(a2 + 48);
  v3 = (__int64 *)(v2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v2 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_10;
  v5 = v2 & 7;
  if ( v5 )
  {
    if ( v5 == 3 )
    {
      v3 = (__int64 *)v3[2];
      goto LABEL_4;
    }
LABEL_10:
    BUG();
  }
  *(_QWORD *)(a2 + 48) = v3;
LABEL_4:
  v6 = *v3;
  if ( !*v3 || (v6 & 4) == 0 )
    BUG();
  v7 = *(_QWORD *)(a1 + 40);
  v16[0] = 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 32LL);
  v9 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _DWORD *))(*(_QWORD *)v7 + 224LL))(
         v7,
         v8,
         *(unsigned int *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 16),
         v16);
  v10 = *(_QWORD *)(a1 + 408);
  LODWORD(v17) = v16[0];
  *((_QWORD *)&v15 + 1) = v9;
  *(_QWORD *)&v15 = v17;
  return sub_37C6BA0(v10, v8, v11, v16[0], v12, v13, v15, v11);
}
