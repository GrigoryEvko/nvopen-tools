// Function: sub_A849A0
// Address: 0xa849a0
//
__int64 __fastcall sub_A849A0(__int64 a1)
{
  unsigned __int8 v2; // cl
  unsigned int v3; // eax
  __int64 v4; // r13
  __int64 v5; // rdi
  _QWORD *v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned __int8 v17; // al
  __int64 v18; // r13
  _QWORD v19[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  __int64 v21; // [rsp+18h] [rbp-38h]
  __int64 v22; // [rsp+20h] [rbp-30h]
  __int64 v23; // [rsp+28h] [rbp-28h]

  v2 = *(_BYTE *)(a1 - 16);
  if ( (v2 & 2) == 0 )
  {
    v3 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
    if ( v3 )
    {
      v4 = a1 - 16;
      v5 = a1 - 16 - 8LL * ((v2 >> 2) & 0xF);
      goto LABEL_4;
    }
    return a1;
  }
  v3 = *(_DWORD *)(a1 - 24);
  if ( !v3 )
    return a1;
  v4 = a1 - 16;
  v5 = *(_QWORD *)(a1 - 32);
LABEL_4:
  if ( (unsigned __int8)(**(_BYTE **)v5 - 5) <= 0x1Fu && v3 > 2 )
    return a1;
  v6 = (_QWORD *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  v7 = *(_QWORD *)(a1 + 8) & 4LL;
  if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
    v6 = (_QWORD *)*v6;
  if ( v3 == 3 )
  {
    if ( (v2 & 2) != 0 )
      v19[0] = **(_QWORD **)(a1 - 32);
    else
      v19[0] = *(_QWORD *)(v4 - 8LL * ((v2 >> 2) & 0xF));
    v19[1] = *(_QWORD *)(v5 + 8);
    v20 = sub_B9C770(v6, v19, 2, 0, 1);
    v21 = v20;
    v13 = sub_BCB2E0(v6);
    v14 = sub_AD6530(v13);
    v22 = sub_B98A20(v14, v19, v15, v16);
    v17 = *(_BYTE *)(a1 - 16);
    if ( (v17 & 2) != 0 )
      v18 = *(_QWORD *)(a1 - 32);
    else
      v18 = v4 - 8LL * ((v17 >> 2) & 0xF);
    v23 = *(_QWORD *)(v18 + 16);
    return sub_B9C770(v6, &v20, 4, 0, 1);
  }
  else
  {
    v20 = a1;
    v21 = a1;
    v8 = sub_BCB2E0(v6);
    v9 = sub_AD6530(v8);
    v22 = sub_B98A20(v9, v7, v10, v11);
    return sub_B9C770(v6, &v20, 3, 0, 1);
  }
}
