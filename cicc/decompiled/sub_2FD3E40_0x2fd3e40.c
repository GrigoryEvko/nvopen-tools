// Function: sub_2FD3E40
// Address: 0x2fd3e40
//
__int64 __fastcall sub_2FD3E40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // r9
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 (*v14)(void); // rsi
  __int64 v15; // rax
  __int64 v16; // rcx
  _BYTE *v17; // rsi
  _BYTE *v18; // rdx
  __int64 **v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  void **v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 **v41; // rsi
  __int64 v42; // [rsp+0h] [rbp-16C0h]
  __int64 v43; // [rsp+10h] [rbp-16B0h] BYREF
  void **v44; // [rsp+18h] [rbp-16A8h]
  unsigned int v45; // [rsp+20h] [rbp-16A0h]
  unsigned int v46; // [rsp+24h] [rbp-169Ch]
  char v47; // [rsp+2Ch] [rbp-1694h]
  char v48[16]; // [rsp+30h] [rbp-1690h] BYREF
  char v49[8]; // [rsp+40h] [rbp-1680h] BYREF
  unsigned __int64 v50; // [rsp+48h] [rbp-1678h]
  int v51; // [rsp+54h] [rbp-166Ch]
  int v52; // [rsp+58h] [rbp-1668h]
  char v53; // [rsp+5Ch] [rbp-1664h]
  char v54[16]; // [rsp+60h] [rbp-1660h] BYREF
  _QWORD v55[10]; // [rsp+70h] [rbp-1650h] BYREF
  char v56; // [rsp+C0h] [rbp-1600h] BYREF
  char *v57; // [rsp+5C0h] [rbp-1100h]
  __int64 v58; // [rsp+5C8h] [rbp-10F8h]
  __int64 v59; // [rsp+5D0h] [rbp-10F0h]
  char v60; // [rsp+5D8h] [rbp-10E8h] BYREF
  char *v61; // [rsp+5E8h] [rbp-10D8h]
  __int64 v62; // [rsp+5F0h] [rbp-10D0h]
  char v63; // [rsp+5F8h] [rbp-10C8h] BYREF
  char *v64; // [rsp+638h] [rbp-1088h]
  __int64 v65; // [rsp+640h] [rbp-1080h]
  char v66; // [rsp+648h] [rbp-1078h] BYREF
  int *v67; // [rsp+6D8h] [rbp-FE8h]
  __int64 v68; // [rsp+6E0h] [rbp-FE0h]
  int v69; // [rsp+6E8h] [rbp-FD8h] BYREF
  _BYTE *v70; // [rsp+6F0h] [rbp-FD0h]
  __int64 v71; // [rsp+6F8h] [rbp-FC8h]
  _BYTE v72[144]; // [rsp+700h] [rbp-FC0h] BYREF
  __int64 v73; // [rsp+790h] [rbp-F30h]
  __int64 v74; // [rsp+798h] [rbp-F28h]
  __int64 v75; // [rsp+7A0h] [rbp-F20h]
  char *v76; // [rsp+7A8h] [rbp-F18h]
  __int64 v77; // [rsp+7B0h] [rbp-F10h]
  char v78; // [rsp+7B8h] [rbp-F08h] BYREF
  _QWORD *v79; // [rsp+7D8h] [rbp-EE8h]
  __int64 v80; // [rsp+7E0h] [rbp-EE0h]
  _QWORD v81[4]; // [rsp+7E8h] [rbp-ED8h] BYREF
  _BYTE v82[3768]; // [rsp+808h] [rbp-EB8h] BYREF

  v7 = sub_2EB2140(a4, &qword_501EB00, a3);
  v8 = sub_2EB2140(a4, (__int64 *)&unk_501EC10, a3) + 8;
  v9 = sub_2EB2140(a4, &qword_5025C20, a3);
  v12 = *(_QWORD *)(a3 + 16);
  v13 = v9 + 8;
  v55[0] = *(_QWORD *)(a3 + 48);
  v14 = *(__int64 (**)(void))(*(_QWORD *)v12 + 128LL);
  v15 = 0;
  if ( v14 != sub_2DAC790 )
  {
    v42 = v13;
    v15 = v14();
    v13 = v42;
  }
  v55[4] = v13;
  v16 = 0x400000000LL;
  v57 = &v60;
  v61 = &v63;
  v55[1] = v15;
  v64 = &v66;
  v65 = 0x200000000LL;
  v71 = 0x200000000LL;
  v55[2] = v7 + 8;
  v76 = &v78;
  v55[3] = v8;
  v55[8] = &v56;
  v67 = &v69;
  v17 = v72;
  v79 = v81;
  v18 = v82;
  v55[9] = 0x1000000000LL;
  v62 = 0x1000000000LL;
  memset(&v55[5], 0, 24);
  v58 = 0;
  v59 = 16;
  v68 = 0x200000001LL;
  v69 = -1;
  v70 = v72;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v77 = 0x400000000LL;
  v80 = 0;
  v81[0] = 0;
  v81[1] = 1;
  v81[2] = v82;
  v81[3] = 0x1000000000LL;
  if ( !*(_DWORD *)(v7 + 136)
    || *(_BYTE *)(a3 + 341)
    || (v17 = (_BYTE *)a3, !(unsigned __int8)sub_2FD2820((__int64)v55, a3, (__int64)v82, 0x400000000LL, v10, v11)) )
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    goto LABEL_6;
  }
  sub_2EAFFB0((__int64)&v43);
  if ( v51 == v52 )
  {
    if ( v47 )
    {
      v24 = v44;
      v41 = (__int64 **)&v44[v46];
      v21 = v46;
      v20 = (__int64 **)v44;
      if ( v44 != (void **)v41 )
      {
        while ( *v20 != &qword_4F82400 )
        {
          if ( v41 == ++v20 )
          {
LABEL_13:
            while ( *v24 != &unk_4F82408 )
            {
              if ( ++v24 == (void **)v20 )
                goto LABEL_18;
            }
            goto LABEL_14;
          }
        }
        goto LABEL_14;
      }
      goto LABEL_18;
    }
    if ( sub_C8CA60((__int64)&v43, (__int64)&qword_4F82400) )
      goto LABEL_14;
  }
  if ( !v47 )
  {
LABEL_20:
    sub_C8CC70((__int64)&v43, (__int64)&unk_4F82408, (__int64)v20, v21, v22, v23);
    goto LABEL_14;
  }
  v24 = v44;
  v21 = v46;
  v20 = (__int64 **)&v44[v46];
  if ( v20 != (__int64 **)v44 )
    goto LABEL_13;
LABEL_18:
  if ( (unsigned int)v21 >= v45 )
    goto LABEL_20;
  v21 = (unsigned int)(v21 + 1);
  v46 = v21;
  *v20 = (__int64 *)&unk_4F82408;
  ++v43;
LABEL_14:
  sub_2FCF5C0((__int64)&v43, (__int64)&qword_5025C20, (__int64)v20, v21, v22, v23);
  sub_2FCF5C0((__int64)&v43, (__int64)&unk_501EC10, v25, v26, v27, v28);
  sub_2FCF5C0((__int64)&v43, (__int64)qword_501FE48, v29, v30, v31, v32);
  sub_2FCF5C0((__int64)&v43, (__int64)&unk_501EAD0, v33, v34, v35, v36);
  sub_2FCF5C0((__int64)&v43, (__int64)&qword_501E910, v37, v38, v39, v40);
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v48, (__int64)&v43);
  v17 = (_BYTE *)(a1 + 80);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v54, (__int64)v49);
  if ( !v53 )
    _libc_free(v50);
  if ( !v47 )
    _libc_free((unsigned __int64)v44);
LABEL_6:
  sub_2FCFFF0((__int64)v55, (__int64)v17, (__int64)v18, v16, v10, v11);
  return a1;
}
