// Function: sub_2F48CE0
// Address: 0x2f48ce0
//
__int64 __fastcall sub_2F48CE0(__int64 a1, __int64 a2, unsigned int a3, unsigned __int32 a4, __int64 a5, __int64 a6)
{
  __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rbx
  unsigned __int16 v15; // r10
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rsi
  _QWORD *v19; // rax
  __int64 v20; // r10
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rax
  unsigned int v24; // ecx
  __int16 *v25; // rdx
  int v26; // esi
  __int64 v27; // r9
  __int32 v28; // r9d
  __int64 v29; // rax
  __int64 *v30; // r11
  unsigned __int8 *v31; // rsi
  __int64 v32; // rcx
  __int64 v33; // rcx
  __int64 v34; // rdi
  _QWORD *v35; // rsi
  __int32 v36; // eax
  __int64 v37; // rdx
  __int64 *v38; // [rsp+0h] [rbp-B0h]
  __int64 v39; // [rsp+8h] [rbp-A8h]
  unsigned int v40; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v41; // [rsp+28h] [rbp-88h] BYREF
  __int64 v42[4]; // [rsp+30h] [rbp-80h] BYREF
  __m128i v43; // [rsp+50h] [rbp-60h] BYREF
  __int64 v44; // [rsp+60h] [rbp-50h]
  __int64 v45; // [rsp+68h] [rbp-48h]
  __int64 v46; // [rsp+70h] [rbp-40h]

  if ( *(_QWORD *)(a1 + 368) )
  {
    v43.m128i_i32[0] = a4;
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, _QWORD, __m128i *))(a1 + 376))(
            a1 + 352,
            *(_QWORD *)(a1 + 16),
            *(_QWORD *)(a1 + 8),
            &v43) )
      return 0;
  }
  v11 = *(unsigned int *)(a1 + 424);
  v12 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 624) + 2LL * (a4 & 0x7FFFFFFF));
  if ( v12 < (unsigned int)v11 )
  {
    v13 = *(_QWORD *)(a1 + 416);
    while ( 1 )
    {
      v14 = v13 + 24LL * v12;
      if ( (a4 & 0x7FFFFFFF) == (*(_DWORD *)(v14 + 8) & 0x7FFFFFFF) )
        break;
      v12 += 0x10000;
      if ( (unsigned int)v11 <= v12 )
        return sub_2F47B00(a1, a2, a3, a4, 1u, a6);
    }
    if ( v14 != v13 + 24 * v11 )
    {
      v15 = *(_WORD *)(v14 + 12);
      if ( v15 )
      {
        v17 = *(_QWORD **)(a1 + 1176);
        v40 = v15;
        v18 = (__int64)&v17[*(unsigned int *)(a1 + 1184)];
        v19 = sub_2F413E0(v17, v18, v15);
        a6 = v40;
        if ( (_QWORD *)v18 == v19 )
        {
          v21 = *(_QWORD *)(a1 + 16);
          v22 = *(_QWORD *)(v21 + 8);
          v23 = *(_DWORD *)(v22 + 24 * v20 + 16) >> 12;
          v24 = *(_DWORD *)(v22 + 24 * v20 + 16) & 0xFFF;
          v25 = (__int16 *)(*(_QWORD *)(v21 + 56) + 2 * v23);
          while ( v25 )
          {
            if ( *(_DWORD *)(*(_QWORD *)(a1 + 1112) + 4LL * v24) >= *(_DWORD *)(a1 + 1104) )
              goto LABEL_19;
            v26 = *v25++;
            v24 += v26;
            if ( !(_WORD)v26 )
              break;
          }
        }
        else
        {
LABEL_19:
          sub_2F42840(a1, v40);
          *(_WORD *)(v14 + 12) = 0;
          sub_2F44460(a1, (_BYTE *)a2, v14, 0, 1u, v27);
          v28 = v40;
          v29 = a2;
          if ( (*(_BYTE *)a2 & 4) == 0 && (*(_BYTE *)(a2 + 44) & 8) != 0 )
          {
            do
              v29 = *(_QWORD *)(v29 + 8);
            while ( (*(_BYTE *)(v29 + 44) & 8) != 0 );
          }
          v30 = *(__int64 **)(v29 + 8);
          v31 = *(unsigned __int8 **)(a2 + 56);
          v32 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL);
          v41 = v31;
          v33 = v32 - 800;
          if ( v31 )
          {
            v38 = v30;
            v39 = v33;
            sub_B96E90((__int64)&v41, (__int64)v31, 1);
            v28 = v40;
            v33 = v39;
            v42[0] = (__int64)v41;
            v30 = v38;
            if ( v41 )
            {
              sub_B976B0((__int64)&v41, v41, (__int64)v42);
              v30 = v38;
              v41 = 0;
              v33 = v39;
              v28 = v40;
            }
          }
          else
          {
            v42[0] = 0;
          }
          v34 = *(_QWORD *)(a1 + 384);
          v42[1] = 0;
          v42[2] = 0;
          v35 = sub_2F26260(v34, v30, v42, v33, v28);
          v36 = *(unsigned __int16 *)(v14 + 12);
          v43.m128i_i64[0] = 0x40000000;
          v44 = 0;
          v43.m128i_i32[2] = v36;
          v45 = 0;
          v46 = 0;
          sub_2E8EAD0(v37, (__int64)v35, &v43);
          if ( v42[0] )
            sub_B91220((__int64)v42, v42[0]);
          if ( v41 )
            sub_B91220((__int64)&v41, (__int64)v41);
        }
      }
      v16 = *(_QWORD *)(a2 + 32) + 40LL * a3;
      if ( (*(_DWORD *)v16 & 0xFFF00) != 0 && (*(_BYTE *)(v16 + 4) & 1) == 0 )
        *(_QWORD *)v14 = a2;
    }
  }
  return sub_2F47B00(a1, a2, a3, a4, 1u, a6);
}
