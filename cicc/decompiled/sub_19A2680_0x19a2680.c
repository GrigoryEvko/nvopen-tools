// Function: sub_19A2680
// Address: 0x19a2680
//
void __fastcall sub_19A2680(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __m128i a7,
        __m128i a8)
{
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r13
  bool v15; // al
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // r8d
  int v19; // r9d
  int v20; // esi
  __int64 v21; // rax
  unsigned int v22; // ecx
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 *v25; // rdi
  unsigned int v26; // r9d
  __int64 v27; // r8
  int v28; // r8d
  int v29; // r9d
  __int64 v32; // [rsp+28h] [rbp-98h] BYREF
  _QWORD v33[2]; // [rsp+30h] [rbp-90h] BYREF
  char v34; // [rsp+40h] [rbp-80h]
  __int64 v35; // [rsp+48h] [rbp-78h]
  unsigned __int64 v36[2]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE v37[32]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v38; // [rsp+80h] [rbp-40h]
  __int64 v39; // [rsp+88h] [rbp-38h]

  if ( a6 )
    v12 = *(_QWORD *)(a4 + 80);
  else
    v12 = *(_QWORD *)(*(_QWORD *)(a4 + 32) + 8 * a5);
  v13 = *(_QWORD *)(a1 + 8);
  v32 = v12;
  v14 = sub_199E590((__int64)&v32, v13, a7, a8);
  v15 = sub_14560B0(v32);
  if ( v14 && !v15 )
  {
    v20 = *(_DWORD *)(a4 + 40);
    v33[0] = *(_QWORD *)a4;
    v33[1] = *(_QWORD *)(a4 + 8);
    v34 = *(_BYTE *)(a4 + 16);
    v35 = *(_QWORD *)(a4 + 24);
    v36[0] = (unsigned __int64)v37;
    v36[1] = 0x400000000LL;
    if ( v20 )
      sub_19930D0((__int64)v36, a4 + 32, v16, v17, v18, v19);
    v21 = *(_QWORD *)(a4 + 80);
    v22 = *(_DWORD *)(a2 + 32);
    v33[0] = v14;
    v23 = *(_QWORD *)(a2 + 720);
    v24 = *(_QWORD *)(a2 + 712);
    v25 = *(__int64 **)(a1 + 32);
    v26 = *(_DWORD *)(a2 + 48);
    v38 = v21;
    v27 = *(_QWORD *)(a2 + 40);
    v39 = *(_QWORD *)(a4 + 88);
    if ( sub_1995490(v25, v24, v23, v22, v27, v26, (__int64)v33) )
    {
      if ( a6 )
        v38 = v32;
      else
        *(_QWORD *)(v36[0] + 8 * a5) = v32;
      sub_19A1660(a1, a2, a3, (__int64)v33, v28, v29);
    }
    if ( (_BYTE *)v36[0] != v37 )
      _libc_free(v36[0]);
  }
}
