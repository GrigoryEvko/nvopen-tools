// Function: sub_257D4A0
// Address: 0x257d4a0
//
__int64 __fastcall sub_257D4A0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edi
  _BYTE *v9; // r13
  signed int v10; // r14d
  __int64 v11; // rdx
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdi
  __int64 (__fastcall *v17)(__int64); // rax
  __int64 v18; // rdi
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rsi
  __int64 v22; // rdx
  void (__fastcall *v23)(__int64, int, int); // rax
  _DWORD *v24; // rdi
  bool (__fastcall *v25)(__int64); // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // [rsp-10h] [rbp-90h]
  __int64 v31; // [rsp-8h] [rbp-88h]
  __int64 v32; // [rsp+0h] [rbp-80h] BYREF
  __int64 v33; // [rsp+8h] [rbp-78h]
  __int64 v34; // [rsp+10h] [rbp-70h]
  char *v35; // [rsp+18h] [rbp-68h] BYREF
  __int64 v36; // [rsp+20h] [rbp-60h]
  char v37; // [rsp+28h] [rbp-58h] BYREF
  __int64 v38; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v39; // [rsp+38h] [rbp-48h] BYREF
  __int64 v40; // [rsp+40h] [rbp-40h]
  _BYTE v41[56]; // [rsp+48h] [rbp-38h] BYREF

  v7 = *a2;
  v8 = *((_DWORD *)a2 + 4);
  v35 = &v37;
  v36 = 0;
  v34 = v7;
  if ( v8 )
  {
    v9 = v41;
    sub_2538240((__int64)&v35, (char **)a2 + 1, a3, a4, a5, a6);
    v10 = **(_DWORD **)a1;
    v39 = v41;
    v40 = 0;
    v38 = v34;
    if ( (_DWORD)v36 )
      sub_2538550((__int64)&v39, (__int64)&v35, v26, v27, v28, v29);
  }
  else
  {
    v9 = v41;
    v10 = **(_DWORD **)a1;
    v38 = v7;
    v39 = v41;
    v40 = 0;
  }
  v32 = sub_254CA10((__int64)&v38, v10);
  v33 = v11;
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  if ( (unsigned __int8)sub_2509800(&v32)
    && (v13 = v32,
        v14 = sub_257C550(*(_QWORD *)(a1 + 8), v32, v33, *(_QWORD *)(a1 + 16), 0, 0, 1),
        v15 = v31,
        (v16 = v14) != 0) )
  {
    v17 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 48LL);
    if ( v17 == sub_2534FF0 )
      v18 = v16 + 88;
    else
      v18 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v17)(v16, v13, v30);
    v19 = *(_QWORD *)(a1 + 24);
    if ( !*(_BYTE *)(v19 + 16) )
    {
      *(_BYTE *)(v19 + 16) = 1;
      *(_QWORD *)v19 = &unk_4A172B8;
      *(_QWORD *)(v19 + 8) = 0x3FF00000000LL;
      v19 = *(_QWORD *)(a1 + 24);
    }
    v20 = *(unsigned int *)(v18 + 8);
    v21 = *(unsigned int *)(v18 + 12);
    v22 = *(_QWORD *)(v18 + 8);
    v23 = *(void (__fastcall **)(__int64, int, int))(*(_QWORD *)v19 + 72LL);
    if ( v23 == sub_2535320 )
      *(_QWORD *)(v19 + 8) &= v22;
    else
      v23(v19, v21, v20);
    v24 = *(_DWORD **)(a1 + 24);
    v25 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v24 + 16LL);
    if ( v25 == sub_2506010 )
      LOBYTE(v9) = v24[3] != 0;
    else
      LODWORD(v9) = ((__int64 (__fastcall *)(_DWORD *, __int64, __int64, __int64, __int64, __int64))v25)(
                      v24,
                      v21,
                      v22,
                      v15,
                      v19,
                      v20);
  }
  else
  {
    LODWORD(v9) = 0;
  }
  if ( v35 != &v37 )
    _libc_free((unsigned __int64)v35);
  return (unsigned int)v9;
}
