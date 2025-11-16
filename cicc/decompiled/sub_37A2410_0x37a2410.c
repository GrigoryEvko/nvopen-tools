// Function: sub_37A2410
// Address: 0x37a2410
//
__int64 __fastcall sub_37A2410(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  _QWORD *v5; // r15
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 v8; // r9
  __int64 v9; // r12
  unsigned __int16 *v10; // rdx
  unsigned int v11; // ecx
  __int64 v12; // r8
  __int64 v13; // r12
  __int128 v15; // [rsp-30h] [rbp-90h]
  unsigned int v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  int v20; // [rsp+28h] [rbp-38h]

  v3 = sub_379AB60(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v4 = *(_QWORD *)(a2 + 80);
  v5 = *(_QWORD **)(a1 + 8);
  v7 = v6;
  v8 = *(_QWORD *)(a2 + 40);
  v9 = v3;
  v10 = (unsigned __int16 *)(*(_QWORD *)(v3 + 48) + 16LL * (unsigned int)v6);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v19 = v4;
  if ( v4 )
  {
    v16 = v11;
    v17 = v8;
    v18 = v12;
    sub_B96E90((__int64)&v19, v4, 1);
    v11 = v16;
    v8 = v17;
    v12 = v18;
  }
  v20 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v15 + 1) = v7;
  *(_QWORD *)&v15 = v9;
  v13 = sub_340F900(v5, 0x9Du, (__int64)&v19, v11, v12, v8, v15, *(_OWORD *)(v8 + 40), *(_OWORD *)(v8 + 80));
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  return v13;
}
