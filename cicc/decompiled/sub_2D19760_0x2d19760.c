// Function: sub_2D19760
// Address: 0x2d19760
//
__int64 __fastcall sub_2D19760(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v7; // zf
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // r12d
  __int64 v12; // rdx
  _BYTE *v14; // rsi
  int v15; // eax
  _BYTE *v16; // rsi
  __int64 v17; // [rsp-8h] [rbp-48h]
  _DWORD v18[4]; // [rsp+Ch] [rbp-34h] BYREF
  _DWORD v19[9]; // [rsp+1Ch] [rbp-24h] BYREF

  v7 = *(_BYTE *)(a1 + 184) == 0;
  v18[0] = a2;
  v19[0] = 0;
  if ( !v7 )
  {
    v8 = *(_QWORD *)(a1 + 192);
    if ( v8 != *(_QWORD *)(a1 + 200) )
      *(_QWORD *)(a1 + 200) = v8;
    v9 = *(_QWORD *)(a1 + 136);
    if ( v9 != *(_QWORD *)(a1 + 144) )
      *(_QWORD *)(a1 + 144) = v9;
    *(_BYTE *)(a1 + 184) = 0;
  }
  v10 = a1 + 216;
  v11 = sub_C55A30(v10, a1, a3, a4, a5, a6, v19);
  v12 = v17;
  if ( (_BYTE)v11 )
    return v11;
  v14 = *(_BYTE **)(a1 + 144);
  if ( v14 == *(_BYTE **)(a1 + 152) )
  {
    v10 = a1 + 136;
    sub_B8BBF0(a1 + 136, v14, v19);
  }
  else
  {
    if ( v14 )
    {
      *(_DWORD *)v14 = v19[0];
      v14 = *(_BYTE **)(a1 + 144);
    }
    *(_QWORD *)(a1 + 144) = v14 + 4;
  }
  v15 = v18[0];
  v16 = *(_BYTE **)(a1 + 200);
  *(_WORD *)(a1 + 14) = v18[0];
  if ( v16 == *(_BYTE **)(a1 + 208) )
  {
    v10 = a1 + 192;
    sub_B8BBF0(a1 + 192, v16, v18);
  }
  else
  {
    if ( v16 )
    {
      *(_DWORD *)v16 = v15;
      v16 = *(_BYTE **)(a1 + 200);
    }
    v16 += 4;
    *(_QWORD *)(a1 + 200) = v16;
  }
  if ( !*(_QWORD *)(a1 + 240) )
    sub_4263D6(v10, v16, v12);
  (*(void (__fastcall **)(__int64, _DWORD *, __int64))(a1 + 248))(a1 + 224, v19, v12);
  return v11;
}
