// Function: sub_3423CC0
// Address: 0x3423cc0
//
__int64 __fastcall sub_3423CC0(
        __int64 a1,
        __int64 a2,
        int a3,
        unsigned __int64 a4,
        int a5,
        __int64 a6,
        unsigned __int64 *a7,
        __int64 a8)
{
  char v10; // bl
  int v11; // esi
  __int64 v12; // rdi
  __int64 v13; // r14
  __int16 v14; // ax
  __int64 v15; // rsi
  int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r13
  __int64 v20; // r9
  unsigned int v21; // eax
  unsigned int v23; // eax
  int v24; // [rsp+8h] [rbp-38h]
  char v25; // [rsp+Ch] [rbp-34h]
  unsigned int v26; // [rsp+Ch] [rbp-34h]

  v10 = a6 & 1;
  v11 = *(_DWORD *)(a2 + 68);
  v12 = *(_QWORD *)(a2 + 48);
  v13 = (unsigned int)(v11 - 1);
  v14 = *(_WORD *)(v12 + 16 * v13);
  if ( v14 == 262 )
  {
    if ( v11 == 1 )
    {
      v24 = -1;
      v10 = 0;
    }
    else
    {
      v15 = (unsigned int)(v11 - 2);
      v16 = -1;
      if ( *(_WORD *)(v12 + 16 * v15) == 1 )
        v16 = v15;
      v24 = v16;
      if ( *(_WORD *)(v12 + 16 * v15) != 1 )
        v10 = 0;
    }
  }
  else if ( v14 == 1 )
  {
    v24 = v11 - 1;
    LODWORD(v13) = -1;
    v10 &= v11 != 0;
  }
  else
  {
    v24 = -1;
    v10 = 0;
    LODWORD(v13) = -1;
  }
  v25 = a6;
  v19 = sub_33EC480(*(_QWORD *)(a1 + 64), a2, ~a3, a4, a5, a6, a7, a8);
  if ( a2 == v19 )
    *(_DWORD *)(a2 + 36) = -1;
  v20 = v25 & 4;
  v21 = *(_DWORD *)(v19 + 68);
  if ( (v25 & 4) != 0 )
  {
    --v21;
    if ( (_DWORD)v13 != -1 && (_DWORD)v13 != v21 )
    {
      v26 = v21;
      sub_34161C0(*(_QWORD *)(a1 + 64), a2, v13, v19, v21);
      sub_3421DB0(v19);
      v21 = v26;
    }
  }
  if ( !v10 || (v23 = v21 - 1, v23 == v24) )
  {
    if ( a2 == v19 )
    {
LABEL_22:
      sub_3421DB0(v19);
      return v19;
    }
  }
  else
  {
    sub_34161C0(*(_QWORD *)(a1 + 64), a2, v24, v19, v23);
    sub_3421DB0(v19);
    if ( a2 == v19 )
      goto LABEL_22;
  }
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v19, v17, v18, v20);
  sub_3421DB0(v19);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  return v19;
}
