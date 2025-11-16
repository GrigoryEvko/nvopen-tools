// Function: sub_28B4710
// Address: 0x28b4710
//
__int64 __fastcall sub_28B4710(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6, __m128i a7)
{
  __int64 v9; // rbx
  __int64 v10; // rax
  char v11; // al
  void *v12; // rsi
  __int64 v14; // [rsp+0h] [rbp-B0h]
  __int64 v15; // [rsp+8h] [rbp-A8h]
  __int64 v16; // [rsp+10h] [rbp-A0h]
  __int64 v17; // [rsp+18h] [rbp-98h]
  __int64 v18; // [rsp+20h] [rbp-90h] BYREF
  _QWORD *v19; // [rsp+28h] [rbp-88h]
  int v20; // [rsp+30h] [rbp-80h]
  int v21; // [rsp+34h] [rbp-7Ch]
  int v22; // [rsp+38h] [rbp-78h]
  char v23; // [rsp+3Ch] [rbp-74h]
  _QWORD v24[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v25; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v26; // [rsp+58h] [rbp-58h]
  __int64 v27; // [rsp+60h] [rbp-50h]
  int v28; // [rsp+68h] [rbp-48h]
  char v29; // [rsp+6Ch] [rbp-44h]
  _BYTE v30[64]; // [rsp+70h] [rbp-40h] BYREF

  v15 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v16 = sub_BC1CD0(a4, &unk_4F86540, a3);
  v9 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v14 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v17 = sub_BC1CD0(a4, &unk_4F8FBC8, a3);
  v10 = sub_BC1CD0(a4, &unk_4F8F810, a3);
  v11 = sub_28B4370(a2, a3, v15 + 8, v16 + 8, v9 + 8, v14 + 8, a5, a6, a7, v17 + 8, *(_QWORD *)(v10 + 8));
  v12 = (void *)(a1 + 32);
  if ( v11 )
  {
    v19 = v24;
    v20 = 2;
    v22 = 0;
    v23 = 1;
    v25 = 0;
    v26 = v30;
    v27 = 2;
    v28 = 0;
    v29 = 1;
    v21 = 1;
    v24[0] = &unk_4F82408;
    v18 = 1;
    if ( &unk_4F82408 != (_UNKNOWN *)&qword_4F82400 && &unk_4F82408 != &unk_4F8F810 )
    {
      v21 = 2;
      v18 = 2;
      v24[1] = &unk_4F8F810;
    }
    sub_C8CF70(a1, v12, 2, (__int64)v24, (__int64)&v18);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v30, (__int64)&v25);
    if ( !v29 )
      _libc_free((unsigned __int64)v26);
    if ( !v23 )
      _libc_free((unsigned __int64)v19);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v12;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
