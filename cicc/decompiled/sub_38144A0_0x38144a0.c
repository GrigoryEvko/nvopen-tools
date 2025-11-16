// Function: sub_38144A0
// Address: 0x38144a0
//
__int64 *__fastcall sub_38144A0(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r15
  __int128 *v10; // rbx
  int v11; // esi
  __int64 *v12; // r11
  const __m128i *v13; // r9
  __int128 *v14; // rax
  __int64 v15; // r8
  unsigned __int16 v16; // cx
  __int64 v17; // rax
  __int64 *v18; // r15
  int v19; // eax
  unsigned int (*v21)(void); // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int16 v24; // [rsp+0h] [rbp-80h]
  const __m128i *v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 *v27; // [rsp+18h] [rbp-68h]
  __int128 *v28; // [rsp+28h] [rbp-58h]
  __int64 v29; // [rsp+30h] [rbp-50h] BYREF
  int v30; // [rsp+38h] [rbp-48h]
  __int64 v31; // [rsp+40h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v29, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    v8 = (unsigned __int16)v30;
    v9 = v31;
  }
  else
  {
    v8 = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v9 = v23;
  }
  v10 = *(__int128 **)(a2 + 40);
  v11 = *(_DWORD *)(a2 + 24);
  v12 = (__int64 *)a1[1];
  v13 = *(const __m128i **)(a2 + 112);
  v14 = v10 + 5;
  v15 = *(_QWORD *)(a2 + 104);
  if ( v11 != 339 )
    v14 = (__int128 *)((char *)v10 + 40);
  v16 = *(_WORD *)(a2 + 96);
  v28 = v14;
  v17 = *(_QWORD *)(a2 + 80);
  v29 = v17;
  if ( v17 )
  {
    v24 = v16;
    v25 = v13;
    v26 = v15;
    v27 = v12;
    sub_B96E90((__int64)&v29, v17, 1);
    v11 = *(_DWORD *)(a2 + 24);
    v16 = v24;
    v13 = v25;
    v15 = v26;
    v12 = v27;
  }
  v30 = *(_DWORD *)(a2 + 72);
  v18 = sub_33E6F50(v12, v11, (__int64)&v29, v16, v15, v13, v8, v9, *v10, *v28);
  if ( v29 )
    sub_B91220((__int64)&v29, v29);
  if ( *(_DWORD *)(a2 + 24) == 338 )
  {
    LOBYTE(v19) = (*(_BYTE *)(a2 + 33) >> 2) & 3;
    if ( !(_BYTE)v19 )
    {
      v21 = *(unsigned int (**)(void))(*(_QWORD *)*a1 + 1208LL);
      LOBYTE(v19) = 3;
      if ( (char *)v21 != (char *)sub_2FE3350 )
      {
        v22 = v21() - 213;
        if ( (unsigned int)v22 > 2 )
          BUG();
        v19 = dword_452AF38[v22] & 3;
      }
    }
    *((_BYTE *)v18 + 33) = *((_BYTE *)v18 + 33) & 0xF3 | (4 * (v19 & 3));
  }
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v18, 1);
  return v18;
}
