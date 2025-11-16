// Function: sub_3265190
// Address: 0x3265190
//
__int64 __fastcall sub_3265190(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rax
  __int64 v8; // r8
  __int16 v9; // r12
  __int64 v10; // rbx
  unsigned __int64 v11; // r11
  __int64 v12; // rsi
  __int64 v13; // r14
  int v14; // ebx
  int v15; // r12d
  __int64 v16; // r12
  __int64 v18; // rsi
  __int64 v19; // rdx
  int v20; // ecx
  unsigned int v21; // edx
  __int128 v22; // [rsp-10h] [rbp-C0h]
  unsigned __int64 v23; // [rsp+8h] [rbp-A8h]
  __int64 v24; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v25; // [rsp+18h] [rbp-98h]
  __int64 v26; // [rsp+20h] [rbp-90h]
  unsigned __int64 v27; // [rsp+28h] [rbp-88h]
  int v28; // [rsp+30h] [rbp-80h]
  __int64 v29; // [rsp+30h] [rbp-80h]
  unsigned __int64 v30; // [rsp+38h] [rbp-78h]
  __int128 v31; // [rsp+40h] [rbp-70h]
  int v32; // [rsp+50h] [rbp-60h]
  unsigned int v33; // [rsp+58h] [rbp-58h]
  __int64 v34; // [rsp+58h] [rbp-58h]
  __int64 v35; // [rsp+70h] [rbp-40h] BYREF
  int v36; // [rsp+78h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 40);
  v8 = *(_QWORD *)(v7 + 80);
  v9 = (*(_WORD *)(a2 + 32) >> 7) & 7;
  v10 = *(_QWORD *)(v7 + 40);
  v31 = (__int128)_mm_loadu_si128((const __m128i *)(v7 + 40));
  v33 = *(_DWORD *)(v7 + 48);
  v11 = *(_QWORD *)(v7 + 88);
  if ( *(_DWORD *)(v8 + 24) == 35 )
  {
    v18 = *(_QWORD *)(v8 + 80);
    v26 = *a1;
    v19 = *(_QWORD *)(*(_QWORD *)(v8 + 48) + 8LL);
    v20 = **(unsigned __int16 **)(v8 + 48);
    v35 = v18;
    v28 = v19;
    if ( v18 )
    {
      v32 = v20;
      v23 = v11;
      v24 = v8;
      sub_B96E90((__int64)&v35, v18, 1);
      v20 = v32;
      v11 = v23;
      v8 = v24;
    }
    v36 = *(_DWORD *)(v8 + 72);
    v25 = v11;
    v8 = sub_33FF780(v26, *(_QWORD *)(v8 + 96), (unsigned int)&v35, v20, v28, 0, 0);
    v11 = v21 | v25 & 0xFFFFFFFF00000000LL;
    if ( v35 )
    {
      v27 = v21 | v25 & 0xFFFFFFFF00000000LL;
      v29 = v8;
      sub_B91220((__int64)&v35, v35);
      v11 = v27;
      v8 = v29;
    }
  }
  v12 = *(_QWORD *)(a2 + 80);
  v13 = *a1;
  v14 = *(unsigned __int16 *)(*(_QWORD *)(v10 + 48) + 16LL * v33);
  v35 = v12;
  v15 = ((v9 & 0xFD) != 1) + 56;
  if ( v12 )
  {
    v30 = v11;
    v34 = v8;
    sub_B96E90((__int64)&v35, v12, 1);
    v11 = v30;
    v8 = v34;
  }
  *((_QWORD *)&v22 + 1) = v11;
  *(_QWORD *)&v22 = v8;
  v36 = *(_DWORD *)(a2 + 72);
  v16 = sub_3406EB0(v13, v15, (unsigned int)&v35, v14, 0, a6, v31, v22);
  if ( v35 )
    sub_B91220((__int64)&v35, v35);
  return v16;
}
