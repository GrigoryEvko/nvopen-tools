// Function: sub_25A1B90
// Address: 0x25a1b90
//
__int64 __fastcall sub_25A1B90(__int64 a1, __int64 a2)
{
  __int16 v3; // ax
  __int64 v4; // rdi
  int v5; // edx
  bool v6; // zf
  __int16 v7; // ax
  __int64 v8; // rdi
  unsigned __int8 v9; // bl
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // r15
  unsigned int v24; // eax
  __int64 v25; // [rsp+8h] [rbp-208h]
  __int64 v26; // [rsp+10h] [rbp-200h]
  __int64 v27; // [rsp+18h] [rbp-1F8h]
  __int64 v28; // [rsp+18h] [rbp-1F8h]
  __int64 v29; // [rsp+20h] [rbp-1F0h] BYREF
  int v30; // [rsp+28h] [rbp-1E8h]
  __int64 v31; // [rsp+30h] [rbp-1E0h] BYREF
  int v32; // [rsp+38h] [rbp-1D8h]
  _QWORD v33[2]; // [rsp+40h] [rbp-1D0h] BYREF
  __int16 v34; // [rsp+50h] [rbp-1C0h]
  __int64 v35; // [rsp+58h] [rbp-1B8h]
  __int64 v36; // [rsp+60h] [rbp-1B0h]
  __int64 v37; // [rsp+68h] [rbp-1A8h]
  __int64 v38; // [rsp+70h] [rbp-1A0h]
  _QWORD v39[2]; // [rsp+78h] [rbp-198h] BYREF
  _BYTE v40[136]; // [rsp+88h] [rbp-188h] BYREF
  __int64 v41; // [rsp+110h] [rbp-100h] BYREF
  void *v42; // [rsp+118h] [rbp-F8h]
  __int16 v43; // [rsp+120h] [rbp-F0h]
  __int64 v44; // [rsp+128h] [rbp-E8h]
  __int64 v45; // [rsp+130h] [rbp-E0h]
  __int64 v46; // [rsp+138h] [rbp-D8h]
  __int64 v47; // [rsp+140h] [rbp-D0h]
  _QWORD v48[2]; // [rsp+148h] [rbp-C8h] BYREF
  _BYTE v49[184]; // [rsp+158h] [rbp-B8h] BYREF

  v35 = 0;
  v36 = 0;
  v37 = 0;
  v33[0] = &unk_4A170B8;
  v3 = *(_WORD *)(a1 + 16);
  v38 = 0;
  v34 = v3;
  v33[1] = &unk_4A16CD8;
  sub_C7D6A0(0, 0, 8);
  v4 = *(unsigned int *)(a1 + 48);
  LODWORD(v38) = v4;
  if ( (_DWORD)v4 )
  {
    v11 = sub_C7D670(16 * v4, 8);
    v12 = (unsigned int)v38;
    v32 = 0;
    v36 = v11;
    v13 = v11;
    v14 = *(_QWORD *)(a1 + 40);
    v31 = -1;
    v37 = v14;
    v15 = *(_QWORD *)(a1 + 32);
    LODWORD(v42) = 0;
    v41 = -2;
    v16 = 0;
    do
    {
      v17 = *(_DWORD *)(v15 + 8);
      *(_DWORD *)(v13 + 8) = v17;
      if ( v17 <= 0x40 )
      {
        *(_QWORD *)v13 = *(_QWORD *)v15;
      }
      else
      {
        v25 = v16;
        v26 = v12;
        v27 = v15;
        sub_C43780(v13, (const void **)v15);
        v16 = v25;
        v12 = v26;
        v15 = v27;
      }
      ++v16;
      v13 += 16;
      v15 += 16;
    }
    while ( v12 != v16 );
    sub_969240(&v41);
    sub_969240(&v31);
  }
  else
  {
    v36 = 0;
    v37 = 0;
  }
  v5 = *(_DWORD *)(a1 + 64);
  v39[0] = v40;
  v39[1] = 0x800000000LL;
  if ( v5 )
    sub_2560D30((unsigned int *)v39, a1 + 56);
  v6 = *(_BYTE *)(a2 + 17) == 0;
  v40[128] = *(_BYTE *)(a1 + 200);
  if ( v6 )
    *(_BYTE *)(a1 + 17) = *(_BYTE *)(a1 + 16);
  sub_2576560(a1, a2);
  v41 = (__int64)&unk_4A170B8;
  v7 = *(_WORD *)(a1 + 16);
  v44 = 0;
  v43 = v7;
  v42 = &unk_4A16CD8;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  sub_C7D6A0(0, 0, 8);
  v8 = *(unsigned int *)(a1 + 48);
  LODWORD(v47) = v8;
  if ( (_DWORD)v8 )
  {
    v18 = sub_C7D670(16 * v8, 8);
    v19 = (unsigned int)v47;
    v20 = *(_QWORD *)(a1 + 32);
    v30 = 0;
    v45 = v18;
    v21 = v18;
    v22 = *(_QWORD *)(a1 + 40);
    v29 = -1;
    v46 = v22;
    v32 = 0;
    v31 = -2;
    if ( (_DWORD)v47 )
    {
      v23 = 0;
      do
      {
        v24 = *(_DWORD *)(v20 + 8);
        *(_DWORD *)(v21 + 8) = v24;
        if ( v24 <= 0x40 )
        {
          *(_QWORD *)v21 = *(_QWORD *)v20;
        }
        else
        {
          v28 = v19;
          sub_C43780(v21, (const void **)v20);
          v19 = v28;
        }
        ++v23;
        v21 += 16;
        v20 += 16;
      }
      while ( v19 != v23 );
    }
    sub_969240(&v31);
    sub_969240(&v29);
  }
  else
  {
    v45 = 0;
    v46 = 0;
  }
  v48[0] = v49;
  v48[1] = 0x800000000LL;
  if ( *(_DWORD *)(a1 + 64) )
    sub_2560D30((unsigned int *)v48, a1 + 56);
  v49[128] = *(_BYTE *)(a1 + 200);
  sub_25485A0((__int64)&v41);
  v9 = sub_255BE50((__int64)v33, (const void ***)a1);
  sub_25485A0((__int64)v33);
  return v9;
}
