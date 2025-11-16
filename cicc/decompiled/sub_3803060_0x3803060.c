// Function: sub_3803060
// Address: 0x3803060
//
__int64 __fastcall sub_3803060(__int64 a1, __int64 a2)
{
  unsigned __int16 *v3; // rax
  __int64 v4; // rsi
  __int64 v5; // r11
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int); // r9
  __int64 v9; // rax
  _WORD *v10; // r10
  __int16 v11; // cx
  int v12; // r11d
  __int64 v13; // r14
  __int64 v15; // [rsp+0h] [rbp-B0h]
  __int16 v16; // [rsp+Ch] [rbp-A4h]
  __int64 (__fastcall *v17)(__int64, __int64, unsigned int); // [rsp+10h] [rbp-A0h]
  _WORD *v18; // [rsp+18h] [rbp-98h]
  __int64 v19; // [rsp+20h] [rbp-90h] BYREF
  int v20; // [rsp+28h] [rbp-88h]
  __int64 v21; // [rsp+30h] [rbp-80h] BYREF
  __int64 v22; // [rsp+50h] [rbp-60h]
  __int64 v23; // [rsp+58h] [rbp-58h]
  __int64 v24; // [rsp+60h] [rbp-50h]
  __int64 v25; // [rsp+68h] [rbp-48h]
  __int64 v26; // [rsp+70h] [rbp-40h]

  v3 = *(unsigned __int16 **)(a2 + 48);
  v4 = *(_QWORD *)(a2 + 40);
  v22 = 0;
  v5 = *(_QWORD *)(a2 + 80);
  v23 = 0;
  v6 = *(_QWORD *)v4;
  v7 = *v3;
  v24 = 0;
  v8 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)v3 + 1);
  v9 = *(unsigned int *)(v4 + 8);
  v25 = 0;
  LOBYTE(v26) = 4;
  v10 = *(_WORD **)a1;
  v11 = *(_WORD *)(*(_QWORD *)(v6 + 48) + 16 * v9);
  v19 = v5;
  if ( v5 )
  {
    v15 = v7;
    v16 = v11;
    v17 = v8;
    v18 = v10;
    sub_B96E90((__int64)&v19, v5, 1);
    v4 = *(_QWORD *)(a2 + 40);
    v7 = v15;
    v11 = v16;
    v8 = v17;
    v10 = v18;
  }
  v12 = 302;
  v20 = *(_DWORD *)(a2 + 72);
  if ( v11 != 12 )
  {
    v12 = 303;
    if ( v11 != 13 )
    {
      v12 = 304;
      if ( v11 != 14 )
      {
        v12 = 305;
        if ( v11 != 15 )
        {
          v12 = 729;
          if ( v11 == 16 )
            v12 = 306;
        }
      }
    }
  }
  sub_3494590(
    (__int64)&v21,
    v10,
    *(_QWORD *)(a1 + 8),
    v12,
    v7,
    v8,
    v4,
    1u,
    v22,
    v23,
    v24,
    v25,
    v26,
    (__int64)&v19,
    0,
    0);
  v13 = v21;
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  return v13;
}
