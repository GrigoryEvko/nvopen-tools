// Function: sub_37A1660
// Address: 0x37a1660
//
__int64 __fastcall sub_37A1660(__int64 *a1, __int64 a2)
{
  __int16 *v3; // rax
  unsigned __int16 v4; // si
  __int64 v5; // r8
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v7; // ecx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // ecx
  __int64 v11; // r10
  _QWORD *v12; // rax
  __int64 v13; // r11
  __int64 v14; // r14
  __int64 v15; // r15
  __int64 v16; // r8
  __int64 v17; // r9
  _QWORD *v18; // rdi
  __int64 v19; // r14
  unsigned int v21; // eax
  __int64 v22; // rdx
  __int128 v23; // [rsp-30h] [rbp-B0h]
  __int128 v24; // [rsp-20h] [rbp-A0h]
  __int128 v25; // [rsp-10h] [rbp-90h]
  __int64 v26; // [rsp+0h] [rbp-80h]
  __int64 v27; // [rsp+8h] [rbp-78h]
  unsigned int v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+18h] [rbp-68h]
  unsigned int v31; // [rsp+20h] [rbp-60h]
  __int64 v32; // [rsp+28h] [rbp-58h]
  __int64 v33; // [rsp+30h] [rbp-50h] BYREF
  int v34; // [rsp+38h] [rbp-48h]
  __int64 v35; // [rsp+40h] [rbp-40h]

  v3 = *(__int16 **)(a2 + 48);
  v4 = *v3;
  v5 = *((_QWORD *)v3 + 1);
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v6 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v33, *a1, *(_QWORD *)(a1[1] + 64), v4, v5);
    v7 = (unsigned __int16)v34;
    v32 = v35;
  }
  else
  {
    v21 = v6(*a1, *(_QWORD *)(a1[1] + 64), v4, v5);
    v32 = v22;
    v7 = v21;
  }
  v28 = v7;
  v8 = sub_379AB60((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v10 = v28;
  v11 = v8;
  v12 = *(_QWORD **)(a2 + 40);
  v13 = v9;
  v14 = v12[5];
  v15 = v12[6];
  v16 = v12[10];
  v17 = v12[11];
  v33 = *(_QWORD *)(a2 + 80);
  if ( v33 )
  {
    v30 = v9;
    v31 = v28;
    v26 = v16;
    v27 = v17;
    v29 = v11;
    sub_B96E90((__int64)&v33, v33, 1);
    v10 = v31;
    v16 = v26;
    v17 = v27;
    v11 = v29;
    v13 = v30;
  }
  *((_QWORD *)&v25 + 1) = v17;
  *(_QWORD *)&v25 = v16;
  v18 = (_QWORD *)a1[1];
  *((_QWORD *)&v24 + 1) = v15;
  *(_QWORD *)&v24 = v14;
  *((_QWORD *)&v23 + 1) = v13;
  *(_QWORD *)&v23 = v11;
  v34 = *(_DWORD *)(a2 + 72);
  v19 = sub_340F900(v18, 0xA0u, (__int64)&v33, v10, v32, v17, v23, v24, v25);
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  return v19;
}
