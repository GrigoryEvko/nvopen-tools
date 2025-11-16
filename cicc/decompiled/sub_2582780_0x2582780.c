// Function: sub_2582780
// Address: 0x2582780
//
__int64 __fastcall sub_2582780(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // r15
  __int16 v4; // ax
  unsigned int v5; // eax
  int v6; // esi
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  unsigned int v10; // r13d
  _BYTE *v11; // r14
  unsigned __int64 v12; // r12
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // ecx
  __int64 v24; // [rsp+8h] [rbp-138h]
  __int64 v25; // [rsp+10h] [rbp-130h]
  __int64 v26; // [rsp+18h] [rbp-128h]
  __int64 v27; // [rsp+20h] [rbp-120h] BYREF
  int v28; // [rsp+28h] [rbp-118h]
  __int64 v29; // [rsp+30h] [rbp-110h] BYREF
  int v30; // [rsp+38h] [rbp-108h]
  _QWORD v31[2]; // [rsp+40h] [rbp-100h] BYREF
  __int16 v32; // [rsp+50h] [rbp-F0h]
  __int64 v33; // [rsp+58h] [rbp-E8h]
  __int64 v34; // [rsp+60h] [rbp-E0h]
  __int64 v35; // [rsp+68h] [rbp-D8h]
  __int64 v36; // [rsp+70h] [rbp-D0h]
  _BYTE *v37; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v38; // [rsp+80h] [rbp-C0h]
  _BYTE v39[184]; // [rsp+88h] [rbp-B8h] BYREF

  v3 = sub_250D070((_QWORD *)(a1 + 72));
  v33 = 0;
  v34 = 0;
  v31[0] = &unk_4A170B8;
  v4 = *(_WORD *)(a1 + 104);
  v35 = 0;
  v32 = v4;
  v36 = 0;
  v31[1] = &unk_4A16CD8;
  sub_C7D6A0(0, 0, 8);
  v5 = *(_DWORD *)(a1 + 136);
  LODWORD(v36) = v5;
  if ( v5 )
  {
    v17 = sub_C7D670(16LL * v5, 8);
    v18 = (unsigned int)v36;
    v19 = 0;
    v28 = 0;
    v34 = v17;
    v20 = v17;
    v21 = *(_QWORD *)(a1 + 128);
    v27 = -1;
    v35 = v21;
    v22 = *(_QWORD *)(a1 + 120);
    v30 = 0;
    v29 = -2;
    do
    {
      v23 = *(_DWORD *)(v22 + 8);
      *(_DWORD *)(v20 + 8) = v23;
      if ( v23 <= 0x40 )
      {
        *(_QWORD *)v20 = *(_QWORD *)v22;
      }
      else
      {
        v24 = v19;
        v25 = v18;
        v26 = v22;
        sub_C43780(v20, (const void **)v22);
        v19 = v24;
        v18 = v25;
        v22 = v26;
      }
      ++v19;
      v20 += 16;
      v22 += 16;
    }
    while ( v18 != v19 );
    sub_969240(&v29);
    sub_969240(&v27);
  }
  else
  {
    v34 = 0;
    v35 = 0;
  }
  v6 = *(_DWORD *)(a1 + 152);
  v37 = v39;
  v38 = 0x800000000LL;
  if ( v6 )
    sub_2560D30((unsigned int *)&v37, a1 + 144);
  v39[128] = *(_BYTE *)(a1 + 288);
  v7 = sub_250D2C0(v3, 0);
  v9 = sub_25803A0(a2, v7, v8, a1, 0, 0, 1);
  if ( v9 )
  {
    sub_2576560(a1 + 88, v9 + 88);
    v10 = (unsigned __int8)sub_255BE50((__int64)v31, (const void ***)(a1 + 88));
  }
  else
  {
    v10 = 0;
    *(_BYTE *)(a1 + 105) = *(_BYTE *)(a1 + 104);
  }
  v11 = v37;
  v31[0] = &unk_4A170B8;
  v12 = (unsigned __int64)&v37[16 * (unsigned int)v38];
  if ( v37 != (_BYTE *)v12 )
  {
    do
    {
      v12 -= 16LL;
      if ( *(_DWORD *)(v12 + 8) > 0x40u && *(_QWORD *)v12 )
        j_j___libc_free_0_0(*(_QWORD *)v12);
    }
    while ( v11 != (_BYTE *)v12 );
    v12 = (unsigned __int64)v37;
  }
  if ( (_BYTE *)v12 != v39 )
    _libc_free(v12);
  v13 = (unsigned int)v36;
  if ( (_DWORD)v36 )
  {
    v14 = v34;
    v15 = v34 + 16LL * (unsigned int)v36;
    do
    {
      if ( *(_DWORD *)(v14 + 8) > 0x40u && *(_QWORD *)v14 )
        j_j___libc_free_0_0(*(_QWORD *)v14);
      v14 += 16;
    }
    while ( v15 != v14 );
    v13 = (unsigned int)v36;
  }
  sub_C7D6A0(v34, 16 * v13, 8);
  return v10;
}
