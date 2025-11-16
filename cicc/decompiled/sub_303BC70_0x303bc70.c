// Function: sub_303BC70
// Address: 0x303bc70
//
__int64 __fastcall sub_303BC70(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned __int16 *v7; // rax
  int v8; // edx
  _QWORD *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  unsigned __int16 v12; // cx
  __int64 v13; // rax
  _DWORD *v14; // rdi
  __int64 (*v15)(void); // rax
  _DWORD *v16; // r15
  unsigned int v17; // eax
  unsigned int v18; // edx
  unsigned __int16 v19; // cx
  __int64 result; // rax
  unsigned __int16 v21; // dx
  __int16 v22; // ax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rsi
  int v27; // esi
  int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // rax
  bool v35; // al
  __int64 v36; // rcx
  __int64 v37; // r8
  int v38; // eax
  int v39; // edi
  int v40; // esi
  __int16 v41; // [rsp+Ah] [rbp-86h]
  int v42; // [rsp+20h] [rbp-70h]
  __int64 v43; // [rsp+28h] [rbp-68h]
  __int64 v44; // [rsp+28h] [rbp-68h]
  unsigned int v45; // [rsp+30h] [rbp-60h] BYREF
  __int64 v46; // [rsp+38h] [rbp-58h]
  unsigned __int16 v47; // [rsp+40h] [rbp-50h] BYREF
  __int64 v48; // [rsp+48h] [rbp-48h]
  __int64 v49; // [rsp+50h] [rbp-40h] BYREF
  int v50; // [rsp+58h] [rbp-38h]

  v7 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  v8 = *v7;
  v46 = *((_QWORD *)v7 + 1);
  v9 = *(_QWORD **)(a2 + 40);
  LOWORD(v45) = v8;
  v43 = *v9;
  v10 = v9[1];
  v42 = v10;
  v11 = *(_QWORD *)(*v9 + 48LL) + 16LL * *((unsigned int *)v9 + 2);
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v47 = v12;
  v48 = v13;
  if ( (_WORD)v8 )
  {
    if ( (unsigned __int16)(v8 - 17) <= 0xD3u )
      LOWORD(v8) = word_4456580[v8 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v45) )
      return a2;
    LOWORD(v8) = sub_3009970((__int64)&v45, v10, v31, v32, v33);
  }
  if ( (_WORD)v8 != 10 )
    return a2;
  v14 = *(_DWORD **)(a1 + 537016);
  v15 = *(__int64 (**)(void))(*(_QWORD *)v14 + 144LL);
  if ( (char *)v15 == (char *)sub_3020010 )
  {
    v16 = v14 + 240;
  }
  else
  {
    v34 = v15();
    v14 = *(_DWORD **)(a1 + 537016);
    v16 = (_DWORD *)v34;
  }
  v17 = v14[85];
  if ( v17 <= 0x31F )
    return sub_346D2C0(v16, a2, a4);
  v18 = v14[84];
  if ( v18 <= 0x45 )
    return sub_346D2C0(v16, a2, a4);
  if ( v17 > 0x383 && v18 > 0x4D )
    return a2;
  v19 = v47;
  if ( v47 )
  {
    if ( (unsigned __int16)(v47 - 17) <= 0xD3u )
      v19 = word_4456580[v47 - 1];
  }
  else
  {
    v35 = sub_30070B0((__int64)&v47);
    v21 = 0;
    if ( !v35 )
      goto LABEL_16;
    v19 = sub_3009970((__int64)&v47, v10, 0, v36, v37);
  }
  result = a2;
  if ( v19 == 12 )
    return result;
  v21 = v47;
LABEL_16:
  v22 = v21;
  if ( v21 )
  {
    if ( (unsigned __int16)(v21 - 17) <= 0xD3u )
      v22 = word_4456580[v21 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v47) )
      return sub_346D2C0(v16, a2, a4);
    v22 = sub_3009970((__int64)&v47, v10, v23, v24, v25);
  }
  if ( v22 != 13 )
    return sub_346D2C0(v16, a2, a4);
  v26 = *(_QWORD *)(a2 + 80);
  v49 = v26;
  if ( v26 )
    sub_B96E90((__int64)&v49, v26, 1);
  v50 = *(_DWORD *)(a2 + 72);
  if ( v47 )
  {
    if ( (unsigned __int16)(v47 - 17) > 0xD3u )
    {
LABEL_24:
      v27 = 12;
      v28 = 0;
      goto LABEL_25;
    }
    v40 = word_4456340[v47 - 1];
    if ( (unsigned __int16)(v47 - 176) > 0x34u )
      LOWORD(v38) = sub_2D43050(12, v40);
    else
      LOWORD(v38) = sub_2D43AD0(12, v40);
    v28 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v47) )
      goto LABEL_24;
    v38 = sub_3009490(&v47, 0xCu, 0);
    v41 = HIWORD(v38);
  }
  HIWORD(v39) = v41;
  LOWORD(v39) = v38;
  v27 = v39;
LABEL_25:
  v29 = sub_346C5C0((_DWORD)v16, v27, v28, v43, v42, (unsigned int)&v49, a4);
  result = sub_3406EE0(a4, v29, v30, &v49, v45, v46);
  if ( v49 )
  {
    v44 = result;
    sub_B91220((__int64)&v49, v49);
    return v44;
  }
  return result;
}
