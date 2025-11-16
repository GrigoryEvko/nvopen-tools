// Function: sub_2439090
// Address: 0x2439090
//
void __fastcall sub_2439090(__int64 a1, __int64 a2, unsigned __int8 a3, int a4, __int64 a5, __int64 a6, __int64 a7)
{
  int v7; // eax
  int v11; // edx
  int v12; // r9d
  int v13; // edi
  int v14; // edx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rax
  int v19; // esi
  unsigned int v20; // esi
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // [rsp+8h] [rbp-138h]
  unsigned int v27; // [rsp+14h] [rbp-12Ch]
  unsigned int v28; // [rsp+28h] [rbp-118h]
  __int64 v29; // [rsp+30h] [rbp-110h] BYREF
  __int64 v30; // [rsp+38h] [rbp-108h]
  __int64 v31; // [rsp+40h] [rbp-100h]
  char v32[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v33; // [rsp+70h] [rbp-D0h]
  unsigned __int64 v34[2]; // [rsp+80h] [rbp-C0h] BYREF
  _BYTE v35[32]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v36; // [rsp+B0h] [rbp-90h]
  __int64 v37; // [rsp+B8h] [rbp-88h]
  __int16 v38; // [rsp+C0h] [rbp-80h]
  __int64 v39; // [rsp+C8h] [rbp-78h]
  void **v40; // [rsp+D0h] [rbp-70h]
  void **v41; // [rsp+D8h] [rbp-68h]
  __int64 v42; // [rsp+E0h] [rbp-60h]
  int v43; // [rsp+E8h] [rbp-58h]
  __int16 v44; // [rsp+ECh] [rbp-54h]
  char v45; // [rsp+EEh] [rbp-52h]
  __int64 v46; // [rsp+F0h] [rbp-50h]
  __int64 v47; // [rsp+F8h] [rbp-48h]
  void *v48; // [rsp+100h] [rbp-40h] BYREF
  void *v49; // [rsp+108h] [rbp-38h] BYREF

  v7 = a3;
  v11 = *(unsigned __int8 *)(a1 + 160);
  v12 = *(unsigned __int8 *)(a1 + 173);
  v13 = 0;
  v14 = (v11 << 25) | (v12 << 24);
  if ( (_BYTE)v12 )
    v13 = *(unsigned __int8 *)(a1 + 172) << 16;
  v27 = a4 | v13 | (16 * v7) | v14 | (32 * *(unsigned __int8 *)(a1 + 161));
  if ( *(_BYTE *)(a1 + 163) )
  {
    sub_2438890(v34, a1, a2, a5, a6, a7);
    a5 = v34[0];
  }
  v26 = a5;
  v15 = sub_BD5C60(a5);
  v40 = &v48;
  v39 = v15;
  v38 = 0;
  v34[0] = (unsigned __int64)v35;
  v48 = &unk_49DA100;
  v34[1] = 0x200000000LL;
  v41 = &v49;
  v49 = &unk_49DA0B0;
  v42 = 0;
  v43 = 0;
  v44 = 512;
  v45 = 7;
  v46 = 0;
  v47 = 0;
  v36 = 0;
  v37 = 0;
  sub_D5F1F0((__int64)v34, v26);
  if ( (unsigned int)(*(_DWORD *)(a1 + 56) - 3) > 2
    || *(_DWORD *)(a1 + 88)
    || (*(_QWORD *)(a1 + 96) & 0xFFFF00000000LL) != *(_QWORD *)(a1 + 96) )
  {
    v16 = *(_QWORD *)(a1 + 512);
    v17 = *(_QWORD *)(a1 + 144);
    v33 = 257;
    v29 = v16;
    v30 = a2;
    v18 = sub_AD64C0(v17, v27, 0);
    v19 = -(*(_BYTE *)(a1 + 164) == 0);
    v31 = v18;
    v20 = (v19 & 0xFFFFFFFE) + 192;
  }
  else
  {
    v21 = *(_QWORD *)(a1 + 144);
    v33 = 257;
    v29 = a2;
    v22 = sub_AD64C0(v21, v27, 0);
    v23 = *(_QWORD *)(a1 + 96);
    v24 = *(_QWORD *)(a1 + 152);
    v30 = v22;
    v25 = sub_AD64C0(v24, v23, 0);
    LODWORD(v23) = -(*(_BYTE *)(a1 + 164) == 0);
    v31 = v25;
    v20 = (v23 & 0xFFFFFFFE) + 193;
  }
  sub_B33D10((__int64)v34, v20, 0, 0, (int)&v29, 3, v28, (__int64)v32);
  nullsub_61();
  v48 = &unk_49DA100;
  nullsub_63();
  if ( (_BYTE *)v34[0] != v35 )
    _libc_free(v34[0]);
}
