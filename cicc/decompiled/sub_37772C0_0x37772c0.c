// Function: sub_37772C0
// Address: 0x37772c0
//
unsigned __int8 *__fastcall sub_37772C0(__int64 a1, __int64 a2, __m128i a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r15d
  unsigned __int16 *v8; // rdx
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r8
  __int16 v12; // cx
  unsigned int *v13; // rax
  unsigned __int8 *v14; // r12
  __int64 v15; // rdx
  __int64 v17; // rsi
  __int64 v18; // r14
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // [rsp+0h] [rbp-50h]
  __int64 v22; // [rsp+10h] [rbp-40h] BYREF
  __int64 v23; // [rsp+18h] [rbp-38h]

  v8 = *(unsigned __int16 **)(a2 + 48);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v22) = v9;
  v23 = v10;
  if ( (_WORD)v9 )
  {
    v11 = 0;
    v12 = word_4456580[v9 - 1];
  }
  else
  {
    v19 = sub_3009970((__int64)&v22, a2, v10, a5, a6);
    HIWORD(v6) = HIWORD(v19);
    v12 = v19;
    v11 = v20;
  }
  v13 = *(unsigned int **)(a2 + 40);
  LOWORD(v6) = v12;
  v14 = *(unsigned __int8 **)v13;
  v15 = *(_QWORD *)(*(_QWORD *)v13 + 48LL) + 16LL * v13[2];
  if ( *(_WORD *)v15 != v12 || *(_QWORD *)(v15 + 8) != v11 && !v12 )
  {
    v17 = *(_QWORD *)(a2 + 80);
    v18 = *(_QWORD *)(a1 + 8);
    v22 = v17;
    if ( v17 )
    {
      v21 = v11;
      sub_B96E90((__int64)&v22, v17, 1);
      v11 = v21;
    }
    LODWORD(v23) = *(_DWORD *)(a2 + 72);
    v14 = sub_33FAF80(v18, 216, (__int64)&v22, v6, v11, (unsigned int)&v22, a3);
    if ( v22 )
      sub_B91220((__int64)&v22, v22);
  }
  return v14;
}
