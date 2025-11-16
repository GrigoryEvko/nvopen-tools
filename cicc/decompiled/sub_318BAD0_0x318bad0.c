// Function: sub_318BAD0
// Address: 0x318bad0
//
_QWORD *__fastcall sub_318BAD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  char v9; // r14
  unsigned __int16 v11; // bx
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // r12
  char v15; // cl
  __int64 v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // r9
  __int64 v19; // r14
  __int64 v20; // rbx
  __int64 v21; // r12
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v25; // rax
  char v26; // al
  __int64 v27; // [rsp+0h] [rbp-70h]
  __int64 v28; // [rsp+0h] [rbp-70h]
  char v29; // [rsp+8h] [rbp-68h]
  char v30[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v31; // [rsp+30h] [rbp-40h]

  v9 = a4;
  v11 = a3;
  v12 = sub_318B710(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  v13 = *(_QWORD *)(a1 + 16);
  v14 = v12;
  v15 = v9;
  v16 = *(_QWORD *)(a2 + 16);
  if ( !HIBYTE(v11) )
  {
    v28 = *(_QWORD *)(a2 + 16);
    v25 = sub_AA4E30(*(_QWORD *)(v12 + 48));
    v26 = sub_AE5020(v25, *(_QWORD *)(v13 + 8));
    v16 = v28;
    v15 = v9;
    LOBYTE(v11) = v26;
  }
  v29 = v15;
  v27 = v16;
  v31 = 257;
  v17 = sub_BD2C40(80, unk_3F10A10);
  v19 = (__int64)v17;
  if ( v17 )
    sub_B4D3C0((__int64)v17, v13, v27, v29, v11, v18, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, char *, _QWORD, _QWORD))(**(_QWORD **)(v14 + 88) + 16LL))(
    *(_QWORD *)(v14 + 88),
    v19,
    v30,
    *(_QWORD *)(v14 + 56),
    *(_QWORD *)(v14 + 64));
  v20 = *(_QWORD *)v14;
  v21 = *(_QWORD *)v14 + 16LL * *(unsigned int *)(v14 + 8);
  while ( v21 != v20 )
  {
    v22 = *(_QWORD *)(v20 + 8);
    v23 = *(_DWORD *)v20;
    v20 += 16;
    sub_B99FD0(v19, v23, v22);
  }
  return sub_3189C10(a5, v19);
}
