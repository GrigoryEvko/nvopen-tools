// Function: sub_258C650
// Address: 0x258c650
//
__int64 __fastcall sub_258C650(__int64 a1, __int64 a2)
{
  __int64 v3; // r10
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // rdi
  unsigned __int8 v8; // al
  unsigned __int8 *v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rcx
  _BYTE *v17; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // [rsp-10h] [rbp-160h]
  __int64 v23; // [rsp+0h] [rbp-150h]
  _QWORD v24[2]; // [rsp+10h] [rbp-140h] BYREF
  __int16 v25; // [rsp+20h] [rbp-130h]
  __int64 v26; // [rsp+28h] [rbp-128h]
  __int64 v27; // [rsp+30h] [rbp-120h]
  __int64 v28; // [rsp+38h] [rbp-118h]
  __int64 v29; // [rsp+40h] [rbp-110h]
  _QWORD v30[2]; // [rsp+48h] [rbp-108h] BYREF
  _BYTE v31[192]; // [rsp+58h] [rbp-F8h] BYREF
  char v32; // [rsp+118h] [rbp-38h]

  v32 = 0;
  v3 = *(_QWORD *)(a1 + 144);
  v25 = 256;
  v26 = 0;
  v24[0] = &unk_4A171B8;
  v4 = v3;
  v27 = 0;
  v28 = 0;
  v24[1] = &unk_4A16CD8;
  v30[0] = v31;
  v30[1] = 0x800000000LL;
  v5 = *(unsigned int *)(a1 + 152);
  v29 = 0;
  v6 = v3 + 24 * v5;
  if ( v6 != v3 )
  {
    do
    {
      if ( *(_BYTE *)(v4 + 16) != 1 )
      {
        v7 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
        if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
          v7 = *(_QWORD *)(v7 + 24);
        v8 = *(_BYTE *)v7;
        if ( *(_BYTE *)v7 )
        {
          if ( v8 == 22 )
          {
            v7 = *(_QWORD *)(v7 + 24);
          }
          else if ( v8 <= 0x1Cu )
          {
            v7 = 0;
          }
          else
          {
            v7 = sub_B43CB0(v7);
          }
        }
        sub_258BA20(a1, a2, v24, *(_QWORD *)v4, *(unsigned __int8 **)(v4 + 8), 2, v7);
      }
      v4 += 24;
    }
    while ( v6 != v4 );
  }
  v23 = sub_25096F0((_QWORD *)(a1 + 72));
  v9 = (unsigned __int8 *)sub_2509740((_QWORD *)(a1 + 72));
  v10 = sub_250D070((_QWORD *)(a1 + 72));
  sub_258BA20(a1, a2, v24, v10, v9, 1, v23);
  v11 = *(_QWORD *)(a1 + 120);
  *(_WORD *)(a1 + 104) = v25;
  sub_C7D6A0(v11, 24LL * *(unsigned int *)(a1 + 136), 8);
  v15 = (unsigned int)v29;
  v16 = v22;
  *(_DWORD *)(a1 + 136) = v29;
  if ( (_DWORD)v15 )
  {
    v19 = sub_C7D670(24 * v15, 8);
    v16 = v27;
    *(_QWORD *)(a1 + 120) = v19;
    v12 = v19;
    *(_QWORD *)(a1 + 128) = v28;
    if ( *(_DWORD *)(a1 + 136) )
    {
      v20 = 0;
      v21 = 24LL * *(unsigned int *)(a1 + 136);
      do
      {
        *(__m128i *)(v12 + v20) = _mm_loadu_si128((const __m128i *)(v16 + v20));
        *(_QWORD *)(v12 + v20 + 16) = *(_QWORD *)(v16 + v20 + 16);
        v20 += 24;
      }
      while ( v20 != v21 );
    }
  }
  else
  {
    *(_QWORD *)(a1 + 120) = 0;
    *(_QWORD *)(a1 + 128) = 0;
  }
  sub_2539BB0(a1 + 144, (__int64)v30, v12, v16, v13, v14);
  v17 = (_BYTE *)v30[0];
  *(_BYTE *)(a1 + 352) = v32;
  v24[0] = &unk_4A171B8;
  if ( v17 != v31 )
    _libc_free((unsigned __int64)v17);
  return sub_C7D6A0(v27, 24LL * (unsigned int)v29, 8);
}
