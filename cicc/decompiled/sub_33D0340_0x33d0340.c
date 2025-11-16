// Function: sub_33D0340
// Address: 0x33d0340
//
__int64 __fastcall sub_33D0340(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rbx
  unsigned __int64 v6; // rax
  __int64 *v7; // r14
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v12; // r15
  __int64 v13; // rdx
  unsigned __int16 v14; // cx
  bool v15; // di
  unsigned __int64 v16; // rsi
  unsigned int v17; // esi
  unsigned int v18; // r13d
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  unsigned __int16 v26; // [rsp+Ch] [rbp-64h]
  __int64 v27; // [rsp+18h] [rbp-58h]
  unsigned __int64 v28; // [rsp+20h] [rbp-50h] BYREF
  __int16 v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h]

  LOWORD(v6) = *(_WORD *)a3;
  v7 = *(__int64 **)(a2 + 64);
  if ( *(_WORD *)a3 )
  {
    if ( (unsigned __int16)(v6 - 17) > 0xD3u )
      goto LABEL_3;
    v12 = 0;
    v13 = (unsigned __int16)v6 - 1;
    v14 = word_4456580[v13];
  }
  else
  {
    if ( !sub_30070B0((__int64)a3) )
    {
LABEL_3:
      v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(**(_QWORD **)(a2 + 16) + 592LL);
      if ( v8 == sub_2D56A50 )
      {
        sub_2FE6CC0((__int64)&v28, *(_QWORD *)(a2 + 16), (__int64)v7, *a3, a3[1]);
        LOWORD(v9) = v29;
        v10 = v30;
      }
      else
      {
        v9 = v8(*(_QWORD *)(a2 + 16), (__int64)v7, *(_DWORD *)a3, a3[1]);
        v3 = v9;
      }
      goto LABEL_5;
    }
    v14 = sub_3009970((__int64)a3, a2, v21, v22, v23);
    LOWORD(v6) = *(_WORD *)a3;
    v12 = v24;
    if ( !*(_WORD *)a3 )
    {
      v26 = v14;
      v25 = sub_3007240((__int64)a3);
      v14 = v26;
      v16 = v25;
      v6 = HIDWORD(v25);
      v28 = v16;
      v15 = v6;
      goto LABEL_8;
    }
    v13 = (unsigned __int16)v6 - 1;
  }
  v15 = (unsigned __int16)(v6 - 176) <= 0x34u;
  LODWORD(v16) = word_4456340[v13];
  LOBYTE(v6) = v15;
LABEL_8:
  v17 = (unsigned int)v16 >> 1;
  BYTE4(v27) = v6;
  v18 = v14;
  LODWORD(v27) = v17;
  if ( v15 )
    LOWORD(v9) = sub_2D43AD0(v14, v17);
  else
    LOWORD(v9) = sub_2D43050(v14, v17);
  v10 = 0;
  if ( !(_WORD)v9 )
    LOWORD(v9) = sub_3009450(v7, v18, v12, v27, v19, v20);
LABEL_5:
  LOWORD(v3) = v9;
  *(_WORD *)a1 = v9;
  *(_QWORD *)(a1 + 16) = v3;
  *(_QWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 24) = v10;
  return a1;
}
