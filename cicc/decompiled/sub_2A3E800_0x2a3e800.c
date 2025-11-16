// Function: sub_2A3E800
// Address: 0x2a3e800
//
void __fastcall sub_2A3E800(__int64 a1, __int64 a2, __int64 **a3, __int64 a4, unsigned int a5, unsigned __int64 a6)
{
  __int64 **v6; // r12
  __int64 v7; // rbx
  _BYTE *v8; // rax
  __int64 v9; // r11
  __int64 v10; // r8
  __int64 **v11; // r13
  __int64 v12; // r9
  unsigned __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rdi
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // rdx
  unsigned __int64 v28; // r8
  __int64 **v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // r13
  _QWORD *v32; // rdi
  __int64 v33; // r13
  _QWORD **v34; // rax
  _QWORD *v35; // rdi
  __int64 **v36; // [rsp+18h] [rbp-1F8h]
  unsigned int v37; // [rsp+18h] [rbp-1F8h]
  __int64 v39; // [rsp+20h] [rbp-1F0h]
  __int64 **v40; // [rsp+28h] [rbp-1E8h]
  _BYTE *v41; // [rsp+28h] [rbp-1E8h]
  __int64 v42; // [rsp+28h] [rbp-1E8h]
  __int64 v46; // [rsp+58h] [rbp-1B8h]
  __int64 v47; // [rsp+58h] [rbp-1B8h]
  __int64 v48; // [rsp+68h] [rbp-1A8h]
  __int64 v49[4]; // [rsp+70h] [rbp-1A0h] BYREF
  _QWORD **v50; // [rsp+90h] [rbp-180h] BYREF
  __int64 v51; // [rsp+98h] [rbp-178h]
  __int64 v52; // [rsp+A0h] [rbp-170h]
  __int16 v53; // [rsp+B0h] [rbp-160h]
  _BYTE *v54; // [rsp+C0h] [rbp-150h]
  __int64 v55; // [rsp+C8h] [rbp-148h]
  _BYTE v56[32]; // [rsp+D0h] [rbp-140h] BYREF
  __int64 v57; // [rsp+F0h] [rbp-120h]
  __int64 v58; // [rsp+F8h] [rbp-118h]
  __int16 v59; // [rsp+100h] [rbp-110h]
  __int64 *v60; // [rsp+108h] [rbp-108h]
  void **v61; // [rsp+110h] [rbp-100h]
  void **v62; // [rsp+118h] [rbp-F8h]
  __int64 v63; // [rsp+120h] [rbp-F0h]
  int v64; // [rsp+128h] [rbp-E8h]
  __int16 v65; // [rsp+12Ch] [rbp-E4h]
  char v66; // [rsp+12Eh] [rbp-E2h]
  __int64 v67; // [rsp+130h] [rbp-E0h]
  __int64 v68; // [rsp+138h] [rbp-D8h]
  void *v69; // [rsp+140h] [rbp-D0h] BYREF
  void *v70; // [rsp+148h] [rbp-C8h] BYREF
  __int64 *v71; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v72; // [rsp+158h] [rbp-B8h]
  _BYTE v73[176]; // [rsp+160h] [rbp-B0h] BYREF

  v6 = a3;
  v7 = a1;
  v60 = *a3;
  v61 = &v69;
  v54 = v56;
  v55 = 0x200000000LL;
  v69 = &unk_49DA100;
  v65 = 512;
  v59 = 0;
  v70 = &unk_49DA0B0;
  v62 = &v70;
  v63 = 0;
  v64 = 0;
  v66 = 7;
  v67 = 0;
  v68 = 0;
  v57 = 0;
  v58 = 0;
  v71 = (__int64 *)v73;
  v72 = 0x1000000000LL;
  v8 = sub_BA8CD0((__int64)a3, a1, 0x11u, 1);
  if ( v8 )
  {
    v9 = (__int64)v8;
    v10 = *((_QWORD *)v8 - 4);
    v11 = **(__int64 ****)(*((_QWORD *)v8 + 3) + 16LL);
    if ( v10 )
    {
      v12 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
      v13 = (unsigned int)(v12 + 1);
      if ( HIDWORD(v72) < (unsigned int)v13 )
      {
        v37 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
        v39 = *((_QWORD *)v8 - 4);
        v41 = v8;
        sub_C8D5F0((__int64)&v71, v73, v13, 8u, v10, v12);
        v12 = v37;
        v10 = v39;
        v9 = (__int64)v41;
      }
      if ( (_DWORD)v12 )
      {
        v14 = 32 * v12;
        v36 = v6;
        v15 = 0;
        v16 = v10;
        v17 = v14;
        v18 = (unsigned int)v72;
        v19 = v9;
        v40 = v11;
        do
        {
          if ( (*(_BYTE *)(v16 + 7) & 0x40) != 0 )
            v20 = *(_QWORD *)(v16 - 8);
          else
            v20 = v16 - 32LL * (*(_DWORD *)(v16 + 4) & 0x7FFFFFF);
          v21 = *(_QWORD *)(v20 + v15);
          if ( v18 + 1 > (unsigned __int64)HIDWORD(v72) )
          {
            sub_C8D5F0((__int64)&v71, v73, v18 + 1, 8u, v10, v14);
            v18 = (unsigned int)v72;
          }
          v15 += 32;
          v71[v18] = v21;
          v18 = (unsigned int)(v72 + 1);
          LODWORD(v72) = v72 + 1;
        }
        while ( v17 != v15 );
        v9 = v19;
        v11 = v40;
        v7 = a1;
        v6 = v36;
      }
    }
    sub_B30290(v9);
  }
  else
  {
    v33 = sub_BCE3C0(v60, 0);
    v42 = sub_BCE3C0(*v6, *(_DWORD *)(*(_QWORD *)(a4 + 8) + 8LL) >> 8);
    v34 = (_QWORD **)sub_BCB2D0(v60);
    v35 = *v34;
    v52 = v33;
    v51 = v42;
    v50 = v34;
    v11 = (__int64 **)sub_BD0B90(v35, &v50, 3, 0);
  }
  v22 = sub_BCB2D0(v60);
  v49[0] = sub_ACD640(v22, a5, 0);
  v49[1] = a4;
  v23 = sub_BCE3C0(v60, 0);
  if ( a6 )
    v24 = sub_ADAFB0(a6, v23);
  else
    v24 = sub_AD6530(v23, 0);
  v49[2] = v24;
  v25 = sub_AD24A0(v11, v49, *((unsigned int *)v11 + 3));
  v27 = (unsigned int)v72;
  v28 = (unsigned int)v72 + 1LL;
  if ( v28 > HIDWORD(v72) )
  {
    v47 = v25;
    sub_C8D5F0((__int64)&v71, v73, (unsigned int)v72 + 1LL, 8u, v28, v26);
    v27 = (unsigned int)v72;
    v25 = v47;
  }
  v71[v27] = v25;
  LODWORD(v72) = v72 + 1;
  v29 = (__int64 **)sub_BCD420((__int64 *)v11, (unsigned int)v72);
  v30 = sub_AD1300(v29, v71, (unsigned int)v72);
  BYTE4(v48) = 0;
  v31 = *(_QWORD **)(v30 + 8);
  v46 = v30;
  v53 = 261;
  v50 = (_QWORD **)v7;
  v51 = 17;
  v32 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v32 )
    sub_B30000((__int64)v32, (__int64)v6, v31, 0, 6, v46, (__int64)&v50, 0, 0, v48, 0);
  if ( v71 != (__int64 *)v73 )
    _libc_free((unsigned __int64)v71);
  nullsub_61();
  v69 = &unk_49DA100;
  nullsub_63();
  if ( v54 != v56 )
    _libc_free((unsigned __int64)v54);
}
