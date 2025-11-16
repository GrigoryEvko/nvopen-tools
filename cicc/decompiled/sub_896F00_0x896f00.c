// Function: sub_896F00
// Address: 0x896f00
//
__int64 __fastcall sub_896F00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // r10
  int v7; // r12d
  __int64 v8; // r13
  __int64 result; // rax
  __int64 *v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // r10
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r10
  const __m128i *v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rbx
  __m128i *v25; // rax
  __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+28h] [rbp-38h]
  const __m128i *v32; // [rsp+28h] [rbp-38h]

  v5 = a1;
  v7 = *(_BYTE *)(a3 + 265) & 1;
  if ( *(_BYTE *)(a2 + 80) == 19 )
  {
    v10 = sub_87F3D0(a2);
    if ( v7 )
    {
      v22 = sub_7259C0(12);
      v14 = a1;
      v26 = 0;
      v8 = (__int64)v22;
      *(_QWORD *)(v22[21] + 16LL) = *(_QWORD *)(a1 + 336);
      v10[11] = (__int64)v22;
      *((_DWORD *)v22 + 46) = v22[23] & 0xFF8FFF00 | 0x70000A;
      v22[13] = *(_QWORD *)(a3 + 128);
    }
    else
    {
      v13 = sub_7259C0(*(_BYTE *)(a3 + 264));
      *((_BYTE *)v13 + 143) |= 8u;
      v8 = (__int64)v13;
      *((_BYTE *)v13 + 177) |= 0x10u;
      v14 = a1;
      v10[11] = (__int64)v13;
      v15 = v13[21];
      v26 = v15;
      *(_QWORD *)(v15 + 160) = *(_QWORD *)(a1 + 336);
      if ( *(_QWORD *)&dword_4D04988 == a2 )
        *(_BYTE *)(v15 + 110) |= 1u;
    }
    v27 = v14;
    *(_BYTE *)(v8 + 88) = sub_87D550(a2) & 3 | *(_BYTE *)(v8 + 88) & 0xFC;
    sub_877D80(v8, v10);
    sub_877F10(v8, (__int64)v10, v16, v17, v18, v19);
    v20 = v27;
    if ( (dword_4F07590 || *(char *)(a3 + 160) < 0) && !*(_DWORD *)(v27 + 52) )
    {
      if ( (*(_BYTE *)(a2 + 81) & 0x10) != 0
        && ((v23 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 64) + 168LL) + 152LL)) == 0
         || (*(_BYTE *)(v23 + 29) & 0x20) != 0) )
      {
        *(_DWORD *)(v27 + 52) = 1;
      }
      else
      {
        sub_7365B0(v8, -1);
        v20 = v27;
      }
    }
    v28 = v20;
    *(_BYTE *)(v8 + 88) = (4 * *(_BYTE *)(a3 + 265)) & 0x70 | *(_BYTE *)(v8 + 88) & 0x8F;
    v21 = (const __m128i *)sub_896D70(a2, **(_QWORD **)(v20 + 192), *(_DWORD *)(v20 + 36));
    v5 = v28;
    if ( v7 )
    {
      **(_QWORD **)(v8 + 168) = v21;
      v24 = *(_QWORD *)(v8 + 168);
      *(_QWORD *)(v24 + 8) = sub_72F240(v21);
      *(_QWORD *)(a3 + 176) = v10;
      goto LABEL_3;
    }
    if ( a5 )
    {
      v32 = v21;
      v25 = sub_72F240(*(const __m128i **)(*(_QWORD *)(*(_QWORD *)(a4 + 88) + 168LL) + 168LL));
      v5 = v28;
      *(_QWORD *)(v26 + 168) = v25;
      *(_QWORD *)(v26 + 176) = v32;
    }
    else
    {
      *(_QWORD *)(v26 + 168) = v21;
    }
    *(_QWORD *)(a3 + 176) = v10;
  }
  else
  {
    v8 = *(_QWORD *)(a2 + 88);
    *(_QWORD *)(a3 + 176) = a2;
    if ( v7 )
      goto LABEL_3;
    v10 = (__int64 *)a2;
  }
  v11 = v10[12];
  *(_BYTE *)(v8 + 177) |= 0xA0u;
  v12 = dword_4F04C34;
  *(_QWORD *)(v11 + 80) = a3;
  *(_QWORD *)(v11 + 120) = *(_QWORD *)(qword_4F04C68[0] + 776 * v12 + 224);
  if ( *(_BYTE *)(a2 + 80) == 19 )
  {
    v31 = v5;
    sub_8CCE20(v10, a3);
    v5 = v31;
  }
  if ( dword_4F07590 && !*(_DWORD *)(v5 + 52) )
  {
    sub_75B260(v8, 6u);
    sub_75BF90(v8);
  }
LABEL_3:
  result = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) & 0x80 | *(_BYTE *)(v8 + 143) & 0x7Fu;
  *(_BYTE *)(v8 + 143) = *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 13) & 0x80 | *(_BYTE *)(v8 + 143) & 0x7F;
  return result;
}
