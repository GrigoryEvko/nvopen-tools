// Function: sub_3374510
// Address: 0x3374510
//
void __fastcall sub_3374510(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rax
  _QWORD *v8; // r12
  __int64 v9; // r13
  int v10; // eax
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rsi
  int v14; // edx
  __int64 v15; // rbx
  int v16; // r13d
  __int64 v17; // [rsp+20h] [rbp-30h] BYREF
  int v18; // [rsp+28h] [rbp-28h]

  v6 = a1[108];
  if ( (*(_BYTE *)(*(_QWORD *)v6 + 877LL) & 2) == 0 )
    return;
  if ( *(_QWORD *)(*(_QWORD *)(a2 + 40) + 56LL) != a2 + 24 )
  {
    v7 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v7 )
    {
      if ( *(_BYTE *)(v7 - 24) == 85 )
      {
        v8 = (_QWORD *)(v7 + 48);
        v9 = v7 - 24;
        if ( (unsigned __int8)sub_A73ED0((_QWORD *)(v7 + 48), 36) || (unsigned __int8)sub_B49560(v9, 36) )
        {
          if ( (*(_BYTE *)(*(_QWORD *)a1[108] + 877LL) & 4) != 0 )
            return;
          v10 = sub_B49240(v9);
          if ( (v10 == 354 || v10 == 361)
            && !(unsigned __int8)sub_A747A0(v8, "trap-func-name", 0xEu)
            && !(unsigned __int8)sub_B49590(v9, "trap-func-name", 0xEu) )
          {
            return;
          }
        }
        v6 = a1[108];
      }
    }
  }
  v11 = *((_DWORD *)a1 + 212);
  v12 = *a1;
  v17 = 0;
  v18 = v11;
  if ( v12 )
  {
    if ( &v17 != (__int64 *)(v12 + 48) )
    {
      v13 = *(_QWORD *)(v12 + 48);
      v17 = v13;
      if ( v13 )
        sub_B96E90((__int64)&v17, v13, 1);
    }
  }
  v15 = sub_33FAF80(v6, 331, (unsigned int)&v17, 1, 0, a6);
  v16 = v14;
  if ( v15 )
  {
    nullsub_1875(v15, v6, 0);
    *(_QWORD *)(v6 + 384) = v15;
    *(_DWORD *)(v6 + 392) = v16;
    sub_33E2B60(v6, 0);
  }
  else
  {
    *(_QWORD *)(v6 + 384) = 0;
    *(_DWORD *)(v6 + 392) = v14;
  }
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
}
