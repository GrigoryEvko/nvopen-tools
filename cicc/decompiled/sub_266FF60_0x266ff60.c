// Function: sub_266FF60
// Address: 0x266ff60
//
__int64 __fastcall sub_266FF60(__int64 a1, __int64 a2)
{
  __int16 v4; // ax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdi
  __int16 v10; // ax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdi
  __int64 v15; // rcx
  __int16 v16; // ax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int16 v22; // ax
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdi
  __int16 v28; // ax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rdi
  __int64 result; // rax
  void *v35; // rax
  __int64 v36; // rdx
  const void *v37; // rsi
  void *v38; // rax
  size_t v39; // rdx
  const void *v40; // rsi
  void *v41; // rax
  __int64 v42; // rdx
  const void *v43; // rsi
  void *v44; // rax
  __int64 v45; // rdx
  const void *v46; // rsi
  void *v47; // rax
  __int64 v48; // rdx
  const void *v49; // rsi

  *(_QWORD *)a1 = off_49D3CA8;
  *(_BYTE *)(a1 + 8) = *(_BYTE *)(a2 + 8);
  v4 = *(_WORD *)(a2 + 24);
  *(_QWORD *)(a1 + 32) = 0;
  *(_WORD *)(a1 + 24) = v4;
  *(_QWORD *)(a1 + 16) = off_4A1FB78;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  sub_C7D6A0(0, 0, 8);
  v9 = *(unsigned int *)(a2 + 56);
  *(_DWORD *)(a1 + 56) = v9;
  if ( (_DWORD)v9 )
  {
    v35 = (void *)sub_C7D670(8 * v9, 8);
    v36 = *(unsigned int *)(a1 + 56);
    v37 = *(const void **)(a2 + 40);
    *(_QWORD *)(a1 + 40) = v35;
    *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
    memcpy(v35, v37, 8 * v36);
  }
  else
  {
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 48) = 0;
  }
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  if ( *(_DWORD *)(a2 + 72) )
    sub_266EB10(a1 + 64, a2 + 64, v5, v6, v7, v8);
  v10 = *(_WORD *)(a2 + 88);
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_WORD *)(a1 + 88) = v10;
  *(_QWORD *)(a1 + 80) = off_4A1FBD8;
  *(_QWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 120) = 0;
  sub_C7D6A0(0, 0, 8);
  v14 = *(unsigned int *)(a2 + 120);
  *(_DWORD *)(a1 + 120) = v14;
  if ( (_DWORD)v14 )
  {
    v47 = (void *)sub_C7D670(8 * v14, 8);
    v48 = *(unsigned int *)(a1 + 120);
    v49 = *(const void **)(a2 + 104);
    *(_QWORD *)(a1 + 104) = v47;
    *(_QWORD *)(a1 + 112) = *(_QWORD *)(a2 + 112);
    memcpy(v47, v49, 8 * v48);
  }
  else
  {
    *(_QWORD *)(a1 + 104) = 0;
    *(_QWORD *)(a1 + 112) = 0;
  }
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 128) = a1 + 144;
  v15 = *(unsigned int *)(a2 + 136);
  if ( (_DWORD)v15 )
    sub_266EB10(a1 + 128, a2 + 128, v11, v15, v12, v13);
  v16 = *(_WORD *)(a2 + 152);
  *(_QWORD *)(a1 + 160) = 0;
  *(_WORD *)(a1 + 152) = v16;
  *(_QWORD *)(a1 + 144) = off_4A1FC38;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_DWORD *)(a1 + 184) = 0;
  sub_C7D6A0(0, 0, 8);
  v20 = *(unsigned int *)(a2 + 184);
  *(_DWORD *)(a1 + 184) = v20;
  if ( (_DWORD)v20 )
  {
    v44 = (void *)sub_C7D670(8 * v20, 8);
    v45 = *(unsigned int *)(a1 + 184);
    v46 = *(const void **)(a2 + 168);
    *(_QWORD *)(a1 + 168) = v44;
    *(_QWORD *)(a1 + 176) = *(_QWORD *)(a2 + 176);
    memcpy(v44, v46, 8 * v45);
  }
  else
  {
    *(_QWORD *)(a1 + 168) = 0;
    *(_QWORD *)(a1 + 176) = 0;
  }
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  v21 = *(unsigned int *)(a2 + 200);
  if ( (_DWORD)v21 )
    sub_266EA30(a1 + 192, a2 + 192, v21, v17, v18, v19);
  *(_QWORD *)(a1 + 208) = *(_QWORD *)(a2 + 208);
  *(_QWORD *)(a1 + 216) = *(_QWORD *)(a2 + 216);
  *(_QWORD *)(a1 + 224) = *(_QWORD *)(a2 + 224);
  *(_BYTE *)(a1 + 232) = *(_BYTE *)(a2 + 232);
  v22 = *(_WORD *)(a2 + 248);
  *(_QWORD *)(a1 + 256) = 0;
  *(_WORD *)(a1 + 248) = v22;
  *(_QWORD *)(a1 + 240) = off_4A1FC98;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  *(_DWORD *)(a1 + 280) = 0;
  sub_C7D6A0(0, 0, 8);
  v27 = *(unsigned int *)(a2 + 280);
  *(_DWORD *)(a1 + 280) = v27;
  if ( (_DWORD)v27 )
  {
    v41 = (void *)sub_C7D670(8 * v27, 8);
    v42 = *(unsigned int *)(a1 + 280);
    v43 = *(const void **)(a2 + 264);
    *(_QWORD *)(a1 + 264) = v41;
    *(_QWORD *)(a1 + 272) = *(_QWORD *)(a2 + 272);
    memcpy(v41, v43, 8 * v42);
  }
  else
  {
    *(_QWORD *)(a1 + 264) = 0;
    *(_QWORD *)(a1 + 272) = 0;
  }
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 288) = a1 + 304;
  if ( *(_DWORD *)(a2 + 296) )
    sub_266E950(a1 + 288, a2 + 288, v23, v24, v25, v26);
  v28 = *(_WORD *)(a2 + 312);
  *(_QWORD *)(a1 + 320) = 0;
  *(_WORD *)(a1 + 312) = v28;
  *(_QWORD *)(a1 + 304) = off_4A1FCF8;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_DWORD *)(a1 + 344) = 0;
  sub_C7D6A0(0, 0, 1);
  v33 = *(unsigned int *)(a2 + 344);
  *(_DWORD *)(a1 + 344) = v33;
  if ( (_DWORD)v33 )
  {
    v38 = (void *)sub_C7D670(v33, 1);
    v39 = *(unsigned int *)(a1 + 344);
    v40 = *(const void **)(a2 + 328);
    *(_QWORD *)(a1 + 328) = v38;
    *(_QWORD *)(a1 + 336) = *(_QWORD *)(a2 + 336);
    memcpy(v38, v40, v39);
  }
  else
  {
    *(_QWORD *)(a1 + 328) = 0;
    *(_QWORD *)(a1 + 336) = 0;
  }
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 352) = a1 + 376;
  *(_QWORD *)(a1 + 368) = 0;
  if ( *(_QWORD *)(a2 + 360) )
    sub_266E880(a1 + 352, a2 + 352, v29, v30, v31, v32);
  result = *(unsigned __int8 *)(a2 + 376);
  *(_BYTE *)(a1 + 376) = result;
  return result;
}
