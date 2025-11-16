// Function: sub_380CD80
// Address: 0x380cd80
//
unsigned __int8 *__fastcall sub_380CD80(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r9
  unsigned __int8 *v6; // r14
  __int64 v7; // rdx
  __int64 v8; // r15
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rsi
  _QWORD *v13; // r10
  __int64 v14; // r8
  __int128 v16; // [rsp-10h] [rbp-70h]
  __int64 v17; // [rsp+0h] [rbp-60h]
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  _QWORD *v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  int v22; // [rsp+28h] [rbp-38h]

  v3 = sub_380AAE0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v4 = *(_QWORD *)(a2 + 48);
  v5 = *(_QWORD *)(a2 + 40);
  v6 = (unsigned __int8 *)v3;
  v8 = v7;
  v9 = *(_QWORD *)(v3 + 48);
  if ( *(_WORD *)v4 == *(_WORD *)v9 && (*(_QWORD *)(v4 + 8) == *(_QWORD *)(v9 + 8) || *(_WORD *)v4) )
  {
    v10 = *(_QWORD *)v5;
    v11 = *(_QWORD *)(v5 + 8);
  }
  else
  {
    v12 = *(_QWORD *)(a2 + 80);
    v13 = *(_QWORD **)(a1 + 8);
    v14 = *(unsigned int *)(a2 + 68);
    v21 = v12;
    if ( v12 )
    {
      v17 = v5;
      v18 = v14;
      v19 = v4;
      v20 = v13;
      sub_B96E90((__int64)&v21, v12, 1);
      v5 = v17;
      v14 = v18;
      v4 = v19;
      v13 = v20;
    }
    *((_QWORD *)&v16 + 1) = v8;
    *(_QWORD *)&v16 = v6;
    v22 = *(_DWORD *)(a2 + 72);
    v6 = sub_3411F20(v13, 146, (__int64)&v21, (unsigned int *)v4, v14, v5, *(_OWORD *)v5, v16);
    if ( v21 )
      sub_B91220((__int64)&v21, v21);
    v10 = (unsigned __int64)v6;
    v11 = 1;
  }
  sub_3760E70(a1, a2, 1, v10, v11);
  return v6;
}
