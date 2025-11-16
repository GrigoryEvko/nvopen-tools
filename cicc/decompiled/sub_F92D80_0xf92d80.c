// Function: sub_F92D80
// Address: 0xf92d80
//
__int64 __fastcall sub_F92D80(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // r15
  unsigned int v7; // eax
  __int64 v8; // rdi
  unsigned int v9; // r12d
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rdi
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // rsi
  unsigned int v17; // r14d
  __int64 v18; // rdi
  __int64 v19; // r14
  bool v20; // al
  __int64 v21; // rdx
  __int64 v22; // rdx
  bool v23; // [rsp+7h] [rbp-59h]
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+28h] [rbp-38h]

  v5 = *(_QWORD *)(a1 - 32);
  v6 = *(_QWORD *)(a1 - 64);
  v7 = sub_D22B80(a1);
  if ( !(_BYTE)v7 )
    return 0;
  v8 = *(_QWORD *)(a2 + 40);
  if ( v8 != v5 )
    return 0;
  v9 = v7;
  if ( !sub_AA54C0(v8) )
    return 0;
  v11 = sub_AA5930(v6);
  if ( v12 != v11 )
    return 0;
  v13 = sub_986580(v6);
  if ( v13 )
  {
    if ( (unsigned int)sub_B46E30(v13) )
      return 0;
  }
  v14 = *(_QWORD *)(a2 - 64);
  if ( v6 == v14
    || !sub_AA4F10(v14)
    || (v24 = *(_QWORD *)(a2 + 40), LOBYTE(v15) = sub_F8EA90(v24), v16 = v24, v17 = v15, !(_BYTE)v15) )
  {
    v18 = *(_QWORD *)(a2 - 32);
    if ( v6 != v18 )
    {
      if ( sub_AA4F10(v18) )
      {
        v19 = *(_QWORD *)(a2 + 40);
        v20 = sub_F8EA90(v19);
        if ( v20 )
        {
          v23 = v20;
          v25 = *(_QWORD *)(a2 - 32);
          sub_AA5980(v25, v19, 0);
          sub_AC2B30(a2 - 32, v6);
          if ( a3 )
          {
            v21 = *(_QWORD *)(a2 + 40);
            v28 = v6 & 0xFFFFFFFFFFFFFFFBLL;
            v27 = v21;
            v29 = v21;
            v30 = v25 | 4;
            sub_FFB3D0(a3, &v27, 2);
            return v23;
          }
          return v9;
        }
      }
    }
    return 0;
  }
  v26 = *(_QWORD *)(a2 - 64);
  sub_AA5980(v26, v16, 0);
  sub_AC2B30(a2 - 64, v6);
  if ( a3 )
  {
    v22 = *(_QWORD *)(a2 + 40);
    v28 = v6 & 0xFFFFFFFFFFFFFFFBLL;
    v9 = v17;
    v27 = v22;
    v29 = v22;
    v30 = v26 | 4;
    sub_FFB3D0(a3, &v27, 2);
  }
  return v9;
}
