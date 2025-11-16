// Function: sub_21C2BE0
// Address: 0x21c2be0
//
__int64 __fastcall sub_21C2BE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9,
        unsigned __int8 a10)
{
  unsigned int v10; // r12d
  int v16; // edx
  int v17; // edx
  __int64 v18; // rsi
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rsi
  int v22; // edx
  __int64 *v24; // rdx
  __int64 v25; // r9
  __int64 v26; // rax
  int v27; // ecx
  __int64 v28; // rsi
  unsigned int v29; // edx
  __int64 v30; // rdi
  unsigned __int64 v31; // rax
  int v32; // ecx
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r12
  __int64 *v37; // rsi
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rsi
  int v41; // edx
  _QWORD *v42; // rax
  int v43; // edx
  int v44; // eax
  int v45; // ecx
  __int64 v46; // [rsp+0h] [rbp-B0h]
  __int64 v47; // [rsp+8h] [rbp-A8h]
  __int64 v48; // [rsp+10h] [rbp-A0h]
  __int64 v49; // [rsp+10h] [rbp-A0h]
  unsigned int v50; // [rsp+10h] [rbp-A0h]
  bool v51; // [rsp+1Fh] [rbp-91h]
  _QWORD v52[2]; // [rsp+60h] [rbp-50h] BYREF
  __int64 v53; // [rsp+70h] [rbp-40h] BYREF
  int v54; // [rsp+78h] [rbp-38h]

  v52[0] = a3;
  v16 = *(unsigned __int16 *)(a3 + 24);
  v52[1] = a4;
  LOBYTE(v10) = v16 == 14 || v16 == 36;
  if ( (_BYTE)v10 )
  {
    *(_QWORD *)a5 = sub_1D299D0(*(_QWORD **)(a1 + 272), *(_DWORD *)(a3 + 84), a10, 0, 1);
    *(_DWORD *)(a5 + 8) = v17;
    v18 = *(_QWORD *)(a2 + 72);
    v19 = *(_QWORD *)(a1 + 272);
    v53 = v18;
    if ( v18 )
      sub_1623A60((__int64)&v53, v18, 2);
    v54 = *(_DWORD *)(a2 + 64);
    v20 = sub_1D38BB0(v19, 0, (__int64)&v53, a10, 0, 1, a7, a8, a9, 0);
    v21 = v53;
    *(_QWORD *)a6 = v20;
    *(_DWORD *)(a6 + 8) = v22;
    if ( v21 )
      sub_161E7C0((__int64)&v53, v21);
  }
  else if ( v16 == 52
         && !sub_21C2A00(a1, **(_QWORD **)(a3 + 32), *(_QWORD *)(*(_QWORD *)(a3 + 32) + 8LL), (__int64)v52)
         && ((unsigned int)sub_1700720(*(_QWORD *)(a1 + 464)) || (*(_BYTE *)(v52[0] + 26LL) & 1) == 0) )
  {
    v24 = *(__int64 **)(v52[0] + 32LL);
    v25 = v24[5];
    v51 = *(_WORD *)(v25 + 24) == 10 || *(_WORD *)(v25 + 24) == 32;
    if ( v51 )
    {
      v26 = *v24;
      v27 = *(unsigned __int16 *)(*v24 + 24);
      if ( v27 == 14 || v27 == 36 )
      {
        v49 = v24[5];
        v42 = sub_1D299D0(*(_QWORD **)(a1 + 272), *(_DWORD *)(v26 + 84), a10, 0, 1);
        v25 = v49;
        *(_QWORD *)a5 = v42;
        *(_DWORD *)(a5 + 8) = v43;
      }
      else
      {
        *(_QWORD *)a5 = v26;
        *(_DWORD *)(a5 + 8) = *((_DWORD *)v24 + 2);
      }
      v28 = *(_QWORD *)(v25 + 88);
      v29 = *(_DWORD *)(v28 + 32);
      v30 = 1LL << ((unsigned __int8)v29 - 1);
      v31 = *(_QWORD *)(v28 + 24);
      if ( v29 > 0x40 )
      {
        v46 = *(_QWORD *)(v25 + 88);
        v47 = v25;
        v50 = *(_DWORD *)(v28 + 32);
        if ( (*(_QWORD *)(v31 + 8LL * ((v29 - 1) >> 6)) & v30) != 0 )
          v44 = sub_16A5810(v28 + 24);
        else
          v44 = sub_16A57B0(v28 + 24);
        v28 = v46;
        v25 = v47;
        v29 = v50;
        v32 = v44;
      }
      else if ( (v30 & v31) != 0 )
      {
        v32 = 64;
        v33 = ~(v31 << (64 - (unsigned __int8)v29));
        if ( v33 )
        {
          _BitScanReverse64(&v34, v33);
          v32 = v34 ^ 0x3F;
        }
      }
      else
      {
        v45 = 64;
        if ( v31 )
        {
          _BitScanReverse64(&v31, v31);
          v45 = v31 ^ 0x3F;
        }
        v32 = v29 + v45 - 64;
      }
      v48 = v25;
      if ( v29 + 1 - v32 <= 0x20 )
      {
        v35 = *(_QWORD *)(a2 + 72);
        v36 = *(_QWORD *)(a1 + 272);
        v53 = v35;
        if ( v35 )
        {
          sub_1623A60((__int64)&v53, v35, 2);
          v28 = *(_QWORD *)(v48 + 88);
          v29 = *(_DWORD *)(v28 + 32);
        }
        v54 = *(_DWORD *)(a2 + 64);
        v37 = *(__int64 **)(v28 + 24);
        if ( v29 > 0x40 )
          v38 = *v37;
        else
          v38 = (__int64)((_QWORD)v37 << (64 - (unsigned __int8)v29)) >> (64 - (unsigned __int8)v29);
        v39 = sub_1D38BB0(v36, v38, (__int64)&v53, 5, 0, 1, a7, a8, a9, 0);
        v40 = v53;
        *(_QWORD *)a6 = v39;
        *(_DWORD *)(a6 + 8) = v41;
        if ( v40 )
          sub_161E7C0((__int64)&v53, v40);
        return v51;
      }
    }
  }
  return v10;
}
