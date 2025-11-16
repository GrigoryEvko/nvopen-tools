// Function: sub_25562B0
// Address: 0x25562b0
//
void __fastcall sub_25562B0(unsigned __int8 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int16 v5; // r15
  __int64 v8; // rdi
  char v9; // al
  __int64 v10; // r13
  __int64 v11; // rax
  __int16 v12; // cx
  __int64 v13; // r15
  __int64 v14; // r11
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rcx
  _QWORD *v18; // rax
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // r13d
  __int64 v26; // rax
  __int64 v27; // rbx
  _QWORD *v28; // rax
  __int64 v29; // r9
  _QWORD *v30; // r15
  __int64 v31; // rdx
  __int64 v32; // rbx
  _QWORD *v33; // rax
  __int64 v34; // r8
  __int64 v35; // r9
  _QWORD *v36; // r12
  __int64 v37; // rax
  __int64 v38; // [rsp+28h] [rbp-138h]
  __int16 v39; // [rsp+30h] [rbp-130h]
  __int64 v40; // [rsp+38h] [rbp-128h]
  int v41; // [rsp+38h] [rbp-128h]
  __int16 v42; // [rsp+40h] [rbp-120h]
  __int64 v43; // [rsp+40h] [rbp-120h]
  __int64 v45; // [rsp+60h] [rbp-100h]
  __int64 v46; // [rsp+60h] [rbp-100h]
  _QWORD *v47; // [rsp+68h] [rbp-F8h]
  _QWORD *v48; // [rsp+68h] [rbp-F8h]
  _QWORD *v49; // [rsp+68h] [rbp-F8h]
  _QWORD *v50; // [rsp+68h] [rbp-F8h]
  unsigned __int64 v51; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v52; // [rsp+78h] [rbp-E8h]
  __int16 v53; // [rsp+90h] [rbp-D0h]
  __int64 v54[2]; // [rsp+A0h] [rbp-C0h] BYREF
  _BYTE v55[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v56; // [rsp+D0h] [rbp-90h]
  __int64 v57; // [rsp+D8h] [rbp-88h]
  __int16 v58; // [rsp+E0h] [rbp-80h]
  __int64 v59; // [rsp+E8h] [rbp-78h]
  void **v60; // [rsp+F0h] [rbp-70h]
  void **v61; // [rsp+F8h] [rbp-68h]
  __int64 v62; // [rsp+100h] [rbp-60h]
  int v63; // [rsp+108h] [rbp-58h]
  __int16 v64; // [rsp+10Ch] [rbp-54h]
  char v65; // [rsp+10Eh] [rbp-52h]
  __int64 v66; // [rsp+110h] [rbp-50h]
  __int64 v67; // [rsp+118h] [rbp-48h]
  void *v68; // [rsp+120h] [rbp-40h] BYREF
  void *v69; // [rsp+128h] [rbp-38h] BYREF

  v5 = a1;
  v59 = sub_BD5C60(a3);
  v60 = &v68;
  v61 = &v69;
  v54[1] = 0x200000000LL;
  v64 = 512;
  v68 = &unk_49DA1B0;
  v54[0] = (__int64)v55;
  v58 = 0;
  v69 = &unk_49DA0B0;
  v62 = 0;
  v63 = 0;
  v65 = 7;
  v66 = 0;
  v67 = 0;
  v56 = 0;
  v57 = 0;
  sub_D5F1F0((__int64)v54, a3);
  v8 = sub_B43CC0(a3);
  v9 = *(_BYTE *)(a2 + 8);
  if ( v9 == 15 )
  {
    v10 = sub_AE4AC0(v8, a2);
    v11 = *(unsigned int *)(a2 + 12);
    if ( (_DWORD)v11 )
    {
      v12 = 2 * v5;
      v13 = 0;
      v40 = 8 * v11;
      v42 = v12;
      do
      {
        v14 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + v13);
        v15 = *(_BYTE *)(v10 + 2 * v13 + 32);
        v51 = *(_QWORD *)(v10 + 2 * v13 + 24);
        v45 = v14;
        LOBYTE(v52) = v15;
        v16 = sub_CA1930(&v51);
        v47 = sub_2538D20(a4, v16, v54, v17);
        v53 = 257;
        v18 = sub_BD2C40(80, unk_3F10A14);
        if ( v18 )
        {
          v20 = (__int64)v47;
          v48 = v18;
          sub_B4D230((__int64)v18, v45, v20, (__int64)&v51, a3 + 24, 0);
          v18 = v48;
        }
        *((_WORD *)v18 + 1) = v42 | *((_WORD *)v18 + 1) & 0xFF81;
        v21 = *(unsigned int *)(a5 + 8);
        if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
        {
          v49 = v18;
          sub_C8D5F0(a5, (const void *)(a5 + 16), v21 + 1, 8u, v21 + 1, v19);
          v21 = *(unsigned int *)(a5 + 8);
          v18 = v49;
        }
        v13 += 8;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v21) = v18;
        ++*(_DWORD *)(a5 + 8);
      }
      while ( v40 != v13 );
    }
  }
  else if ( v9 == 16 )
  {
    v46 = *(_QWORD *)(a2 + 24);
    v22 = sub_9208B0(v8, v46);
    v52 = v23;
    v51 = (unsigned __int64)(v22 + 7) >> 3;
    v43 = sub_CA1930(&v51);
    v41 = *(_QWORD *)(a2 + 32);
    if ( v41 )
    {
      v25 = 0;
      v39 = 2 * v5;
      v26 = a3 + 24;
      v27 = 0;
      v38 = v26;
      do
      {
        v50 = sub_2538D20(a4, v27, v54, v24);
        v53 = 257;
        v28 = sub_BD2C40(80, unk_3F10A14);
        v30 = v28;
        if ( v28 )
          sub_B4D230((__int64)v28, v46, (__int64)v50, (__int64)&v51, v38, 0);
        *((_WORD *)v30 + 1) = v39 | *((_WORD *)v30 + 1) & 0xFF81;
        v31 = *(unsigned int *)(a5 + 8);
        if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
        {
          sub_C8D5F0(a5, (const void *)(a5 + 16), v31 + 1, 8u, v31 + 1, v29);
          v31 = *(unsigned int *)(a5 + 8);
        }
        v24 = *(_QWORD *)a5;
        ++v25;
        v27 += v43;
        *(_QWORD *)(*(_QWORD *)a5 + 8 * v31) = v30;
        ++*(_DWORD *)(a5 + 8);
      }
      while ( v41 != v25 );
    }
  }
  else
  {
    v32 = a3 + 24;
    v53 = 257;
    v33 = sub_BD2C40(80, unk_3F10A14);
    v36 = v33;
    if ( v33 )
      sub_B4D230((__int64)v33, a2, a4, (__int64)&v51, v32, 0);
    *((_WORD *)v36 + 1) = *((_WORD *)v36 + 1) & 0xFF81 | (2 * v5);
    v37 = *(unsigned int *)(a5 + 8);
    if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
    {
      sub_C8D5F0(a5, (const void *)(a5 + 16), v37 + 1, 8u, v34, v35);
      v37 = *(unsigned int *)(a5 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a5 + 8 * v37) = v36;
    ++*(_DWORD *)(a5 + 8);
  }
  nullsub_61();
  v68 = &unk_49DA1B0;
  nullsub_63();
  if ( (_BYTE *)v54[0] != v55 )
    _libc_free(v54[0]);
}
