// Function: sub_2D4FA00
// Address: 0x2d4fa00
//
void __fastcall sub_2D4FA00(unsigned __int64 *a1, __int64 a2, int a3)
{
  unsigned __int64 *v3; // r12
  unsigned __int16 v4; // bx
  unsigned int v5; // ebx
  unsigned __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rsi
  __int64 v9; // r15
  __int64 v10; // r9
  __int64 v11; // r15
  _QWORD *v12; // rax
  char v13; // al
  __int64 v14; // r9
  int v15; // r15d
  __int64 v16; // r15
  unsigned int *v17; // r15
  __int64 v18; // rbx
  unsigned int *v19; // r12
  __int64 v20; // rdx
  unsigned int v21; // esi
  __int64 v22; // rbx
  unsigned int *v23; // r12
  unsigned int *v24; // r15
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // [rsp-10h] [rbp-260h]
  unsigned __int64 *v28; // [rsp+8h] [rbp-248h]
  __int64 v29; // [rsp+10h] [rbp-240h]
  __int64 v30; // [rsp+10h] [rbp-240h]
  __int64 v31; // [rsp+10h] [rbp-240h]
  __int64 v32; // [rsp+10h] [rbp-240h]
  __int64 v33; // [rsp+10h] [rbp-240h]
  unsigned int v34; // [rsp+10h] [rbp-240h]
  __int64 v35; // [rsp+10h] [rbp-240h]
  unsigned int v36; // [rsp+10h] [rbp-240h]
  unsigned __int8 v37; // [rsp+23h] [rbp-22Dh]
  _QWORD *v39; // [rsp+28h] [rbp-228h] BYREF
  unsigned int v40; // [rsp+34h] [rbp-21Ch] BYREF
  __int64 v41; // [rsp+38h] [rbp-218h] BYREF
  char v42[32]; // [rsp+40h] [rbp-210h] BYREF
  __int16 v43; // [rsp+60h] [rbp-1F0h]
  _QWORD v44[4]; // [rsp+70h] [rbp-1E0h] BYREF
  __int16 v45; // [rsp+90h] [rbp-1C0h]
  _QWORD v46[4]; // [rsp+A0h] [rbp-1B0h] BYREF
  __int16 v47; // [rsp+C0h] [rbp-190h]
  __int64 v48[2]; // [rsp+D0h] [rbp-180h] BYREF
  __int64 v49; // [rsp+E0h] [rbp-170h]
  __int64 v50; // [rsp+E8h] [rbp-168h]
  unsigned __int8 v51; // [rsp+F0h] [rbp-160h]
  __int64 v52; // [rsp+F8h] [rbp-158h]
  unsigned int *v53; // [rsp+110h] [rbp-140h] BYREF
  unsigned int v54; // [rsp+118h] [rbp-138h]
  char v55; // [rsp+120h] [rbp-130h] BYREF
  __int64 v56; // [rsp+148h] [rbp-108h]
  __int64 v57; // [rsp+150h] [rbp-100h]
  __int64 v58; // [rsp+160h] [rbp-F0h]
  __int64 v59; // [rsp+168h] [rbp-E8h]
  __int64 v60; // [rsp+170h] [rbp-E0h]
  int v61; // [rsp+178h] [rbp-D8h]
  void *v62; // [rsp+190h] [rbp-C0h]
  void *v63; // [rsp+198h] [rbp-B8h]
  __int64 v64[12]; // [rsp+1F0h] [rbp-60h] BYREF

  v3 = a1;
  v39 = (_QWORD *)a2;
  v40 = (*(_WORD *)(a2 + 2) >> 4) & 0x1F;
  if ( v40 - 5 <= 1 || v40 == 3 )
  {
    v12 = (_QWORD *)sub_2D47CC0(a1, a2);
    sub_2D4BB90(a1, v12);
  }
  else
  {
    v4 = *(_WORD *)(a2 + 2);
    v37 = *(_BYTE *)(a2 + 72);
    sub_2D46B10((__int64)&v53, a2, a1[1]);
    v5 = (v4 >> 1) & 7;
    _BitScanReverse64(&v6, 1LL << (*((_WORD *)v39 + 1) >> 9));
    sub_2D44EF0(
      (__int64)v48,
      (__int64)&v53,
      (__int64)v39,
      v39[1],
      *(v39 - 8),
      63 - (v6 ^ 0x3F),
      *(_DWORD *)(*a1 + 96) >> 3);
    v7 = v27;
    v41 = 0;
    if ( v40 <= 2 || v40 == 4 )
    {
      v45 = 257;
      v9 = *(v39 - 4);
      if ( v49 == *(_QWORD *)(v9 + 8) )
      {
        v10 = *(v39 - 4);
      }
      else
      {
        v29 = v49;
        v10 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v58 + 120LL))(
                v58,
                49,
                v9,
                v49,
                257,
                v27);
        if ( !v10 )
        {
          v47 = 257;
          v31 = sub_B51D30(49, v9, v29, (__int64)v46, 0, 0);
          v13 = sub_920620(v31);
          v14 = v31;
          if ( v13 )
          {
            v15 = v61;
            if ( v60 )
            {
              sub_B99FD0(v31, 3u, v60);
              v14 = v31;
            }
            v32 = v14;
            sub_B45150(v14, v15);
            v14 = v32;
          }
          v33 = v14;
          (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v59 + 16LL))(
            v59,
            v14,
            v44,
            v56,
            v57);
          v10 = v33;
          v16 = 4LL * v54;
          if ( v53 != &v53[v16] )
          {
            v34 = v5;
            v17 = &v53[v16];
            v18 = v10;
            v19 = v53;
            do
            {
              v20 = *((_QWORD *)v19 + 1);
              v21 = *v19;
              v19 += 4;
              sub_B99FD0(v18, v21, v20);
            }
            while ( v17 != v19 );
            v10 = v18;
            v3 = a1;
            v5 = v34;
          }
        }
      }
      v43 = 257;
      v11 = v52;
      v44[0] = "ValOperand_Shifted";
      v45 = 259;
      v30 = sub_A82F30(&v53, v10, v48[0], (__int64)v42, 0);
      v7 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v58 + 32LL))(
             v58,
             25,
             v30,
             v11,
             0,
             0);
      if ( !v7 )
      {
        v47 = 257;
        v35 = sub_B504D0(25, v30, v11, (__int64)v46, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v59 + 16LL))(
          v59,
          v35,
          v44,
          v56,
          v57);
        v7 = v35;
        if ( v53 != &v53[4 * v54] )
        {
          v36 = v5;
          v22 = v7;
          v28 = v3;
          v23 = v53;
          v24 = &v53[4 * v54];
          do
          {
            v25 = *((_QWORD *)v23 + 1);
            v26 = *v23;
            v23 += 4;
            sub_B99FD0(v22, v26, v25);
          }
          while ( v24 != v23 );
          v7 = v22;
          v3 = v28;
          v5 = v36;
        }
      }
      v41 = v7;
    }
    v46[3] = v48;
    v46[0] = &v40;
    v46[1] = &v41;
    v46[2] = &v39;
    if ( a3 == 4 )
      v8 = sub_2D460D0(
             (__int64)&v53,
             v48[0],
             v50,
             v51,
             v5,
             v37,
             sub_2D44A40,
             (__int64)v46,
             (void (__fastcall *)(__int64, __int64, __int64, __int64, __int64, _QWORD, _QWORD, _QWORD, _QWORD *, __int64 *, __int64))sub_2D42AF0,
             (__int64)sub_2D45870,
             (__int64)v39);
    else
      v8 = sub_2D46690(v3, (__int64)&v53, v48[0], v50, v5, v7, sub_2D44A40, (__int64)v46);
    if ( v48[0] != v48[1] )
      v8 = sub_2D44750((__int64 *)&v53, v8, v48);
    sub_BD84D0((__int64)v39, v8);
    sub_B43D60(v39);
    sub_B32BF0(v64);
    v62 = &unk_49E5698;
    v63 = &unk_49D94D0;
    nullsub_63();
    nullsub_63();
    if ( v53 != (unsigned int *)&v55 )
      _libc_free((unsigned __int64)v53);
  }
}
