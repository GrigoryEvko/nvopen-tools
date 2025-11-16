// Function: sub_1D49FB0
// Address: 0x1d49fb0
//
__int64 __fastcall sub_1D49FB0(
        __int64 a1,
        __int64 a2,
        __int16 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8)
{
  char v10; // bl
  int v11; // esi
  __int64 v12; // rdi
  __int64 v13; // r14
  char v14; // al
  __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // r13
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v24; // eax
  int v25; // [rsp+8h] [rbp-38h]
  char v26; // [rsp+Ch] [rbp-34h]
  unsigned int v27; // [rsp+Ch] [rbp-34h]

  v10 = a6 & 1;
  v11 = *(_DWORD *)(a2 + 60);
  v12 = *(_QWORD *)(a2 + 40);
  v13 = (unsigned int)(v11 - 1);
  v14 = *(_BYTE *)(v12 + 16 * v13);
  if ( v14 == 111 )
  {
    if ( v11 == 1 )
    {
      v25 = -1;
      v10 = 0;
    }
    else
    {
      v15 = (unsigned int)(v11 - 2);
      v16 = -1;
      if ( *(_BYTE *)(v12 + 16 * v15) == 1 )
        v16 = v15;
      v25 = v16;
      if ( *(_BYTE *)(v12 + 16 * v15) != 1 )
        v10 = 0;
    }
  }
  else if ( v14 == 1 )
  {
    v25 = v11 - 1;
    LODWORD(v13) = -1;
    v10 &= v11 != 0;
  }
  else
  {
    v25 = -1;
    v10 = 0;
    LODWORD(v13) = -1;
  }
  v26 = a6;
  v17 = sub_1D2E3C0(*(_QWORD *)(a1 + 272), a2, ~a3, a4, a5, a6, a7, a8);
  if ( a2 == v17 )
    *(_DWORD *)(a2 + 28) = -1;
  v18 = *(_DWORD *)(v17 + 60);
  if ( (v26 & 4) != 0 )
  {
    --v18;
    if ( (_DWORD)v13 != -1 && (_DWORD)v13 != v18 )
    {
      v27 = v18;
      sub_1D44C70(*(_QWORD *)(a1 + 272), a2, v13, v17, v18);
      sub_1D49010(v17);
      v18 = v27;
    }
  }
  if ( !v10 || (v24 = v18 - 1, v24 == v25) )
  {
    if ( a2 == v17 )
    {
LABEL_22:
      sub_1D49010(v17);
      return v17;
    }
  }
  else
  {
    sub_1D44C70(*(_QWORD *)(a1 + 272), a2, v25, v17, v24);
    sub_1D49010(v17);
    if ( a2 == v17 )
      goto LABEL_22;
  }
  sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v17);
  sub_1D49010(v17);
  sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v19, v20, v21, v22);
  return v17;
}
