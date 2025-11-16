// Function: sub_1178690
// Address: 0x1178690
//
unsigned __int8 *__fastcall sub_1178690(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4, __int64 a5, char a6)
{
  __int64 v6; // r10
  unsigned int **v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // r15
  unsigned __int8 *v13; // r12
  unsigned __int8 v14; // bl
  unsigned __int8 v15; // al
  int v16; // r13d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // r15
  __int64 v22; // rdx
  unsigned int v23; // esi
  int v24; // [rsp+Ch] [rbp-B4h]
  unsigned __int8 *v25; // [rsp+18h] [rbp-A8h]
  unsigned int v26; // [rsp+20h] [rbp-A0h]
  _BYTE v27[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v28; // [rsp+50h] [rbp-70h]
  _BYTE v29[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v30; // [rsp+80h] [rbp-40h]

  v6 = a2;
  v30 = 257;
  v10 = *(unsigned int ***)a1;
  if ( !a6 )
  {
    v11 = a3;
    a3 = a2;
    v6 = v11;
  }
  v25 = (unsigned __int8 *)sub_B36550(
                             v10,
                             *(_QWORD *)(*(_QWORD *)(a1 + 8) - 96LL),
                             a3,
                             v6,
                             (__int64)v29,
                             *(_QWORD *)(a1 + 8));
  sub_BD6B90(v25, *(unsigned __int8 **)(a1 + 8));
  v12 = *(_QWORD *)a1;
  v28 = 257;
  if ( *(_BYTE *)(v12 + 108) )
  {
    v13 = (unsigned __int8 *)sub_B35400(v12, 0x66u, (__int64)v25, a5, v26, (__int64)v27, 0, 0, 0);
  }
  else
  {
    v13 = (unsigned __int8 *)(*(__int64 (__fastcall **)(_QWORD, __int64, unsigned __int8 *, __int64, _QWORD))(**(_QWORD **)(v12 + 80) + 40LL))(
                               *(_QWORD *)(v12 + 80),
                               14,
                               v25,
                               a5,
                               *(unsigned int *)(v12 + 104));
    if ( !v13 )
    {
      v24 = *(_DWORD *)(v12 + 104);
      v30 = 257;
      v18 = sub_B504D0(14, (__int64)v25, a5, (__int64)v29, 0, 0);
      v19 = *(_QWORD *)(v12 + 96);
      v13 = (unsigned __int8 *)v18;
      if ( v19 )
        sub_B99FD0(v18, 3u, v19);
      sub_B45150((__int64)v13, v24);
      (*(void (__fastcall **)(_QWORD, unsigned __int8 *, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(v12 + 88) + 16LL))(
        *(_QWORD *)(v12 + 88),
        v13,
        v27,
        *(_QWORD *)(v12 + 56),
        *(_QWORD *)(v12 + 64));
      v20 = *(_QWORD *)v12;
      v21 = *(_QWORD *)v12 + 16LL * *(unsigned int *)(v12 + 8);
      while ( v21 != v20 )
      {
        v22 = *(_QWORD *)(v20 + 8);
        v23 = *(_DWORD *)v20;
        v20 += 16;
        sub_B99FD0((__int64)v13, v23, v22);
      }
    }
  }
  sub_BD6B90(v13, a4);
  v14 = sub_B45210(*(_QWORD *)(a1 + 8));
  v15 = sub_B45210((__int64)a4);
  v16 = v15 & v14 & 0x71 | (v15 | v14) & 0xE;
  sub_B45150((__int64)v13, v16);
  sub_B45150((__int64)v25, v16);
  return v13;
}
