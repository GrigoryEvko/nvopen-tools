// Function: sub_318B980
// Address: 0x318b980
//
_QWORD *__fastcall sub_318B980(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned __int16 v10; // r14
  char v12; // bl
  __int64 v13; // rax
  char v14; // r9
  __int64 v15; // r12
  char v16; // r8
  __int64 v17; // rdx
  __int64 v18; // rbx
  _QWORD *v19; // rax
  __int64 v20; // r14
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v26; // rax
  char v27; // al
  char v28; // [rsp+Ch] [rbp-74h]
  __int64 v29; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  char v31; // [rsp+18h] [rbp-68h]
  char v32; // [rsp+18h] [rbp-68h]
  char v33[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v34; // [rsp+40h] [rbp-40h]

  v10 = a3;
  v12 = a4;
  v13 = sub_318B710((__int64)a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v14 = v10;
  v15 = v13;
  v16 = v12;
  v17 = *(_QWORD *)(a2 + 16);
  v18 = *a1;
  if ( !HIBYTE(v10) )
  {
    v30 = *(_QWORD *)(a2 + 16);
    v32 = v16;
    v26 = sub_AA4E30(*(_QWORD *)(v13 + 48));
    v27 = sub_AE5020(v26, v18);
    v17 = v30;
    v16 = v32;
    v14 = v27;
  }
  v28 = v14;
  v29 = v17;
  v31 = v16;
  v34 = 257;
  v19 = sub_BD2C40(80, 1u);
  v20 = (__int64)v19;
  if ( v19 )
    sub_B4D190((__int64)v19, v18, v29, (__int64)v33, v31, v28, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, _QWORD))(**(_QWORD **)(v15 + 88) + 16LL))(
    *(_QWORD *)(v15 + 88),
    v20,
    a6,
    *(_QWORD *)(v15 + 56),
    *(_QWORD *)(v15 + 64));
  v21 = *(_QWORD *)v15;
  v22 = *(_QWORD *)v15 + 16LL * *(unsigned int *)(v15 + 8);
  while ( v22 != v21 )
  {
    v23 = *(_QWORD *)(v21 + 8);
    v24 = *(_DWORD *)v21;
    v21 += 16;
    sub_B99FD0(v20, v24, v23);
  }
  return sub_3189B90(a5, v20);
}
