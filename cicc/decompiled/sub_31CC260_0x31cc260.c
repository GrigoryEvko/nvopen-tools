// Function: sub_31CC260
// Address: 0x31cc260
//
void __fastcall sub_31CC260(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi
  __int64 v3; // rdx
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 *v6; // rdx
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // r13
  char v10; // dl
  __int64 *v11; // rax
  __int64 v12; // rsi
  __int64 (__fastcall *v13)(__int64, unsigned __int64, __int64); // rax
  unsigned int *v14; // r13
  unsigned int *v15; // r14
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // rax
  __int64 v19; // rax
  bool v20; // al
  __int64 v21; // r11
  int v22; // edx
  __int64 v23; // rdx
  __int64 v24; // r9
  _QWORD *v25; // r14
  __int64 v26; // r8
  _QWORD *v27; // rax
  int v28; // edx
  _BYTE *v29; // rcx
  __int64 v30; // r12
  unsigned int *v31; // r13
  unsigned int *v32; // r14
  __int64 v33; // rdx
  unsigned int v34; // esi
  __int64 v35; // [rsp+10h] [rbp-390h]
  int v36; // [rsp+18h] [rbp-388h]
  __int64 v37; // [rsp+20h] [rbp-380h]
  unsigned __int64 v38; // [rsp+20h] [rbp-380h]
  __int64 v39[2]; // [rsp+30h] [rbp-370h] BYREF
  __int64 v40; // [rsp+40h] [rbp-360h] BYREF
  __int64 v41; // [rsp+48h] [rbp-358h]
  __int64 v42; // [rsp+50h] [rbp-350h]
  __int64 v43; // [rsp+58h] [rbp-348h]
  _BYTE v44[32]; // [rsp+60h] [rbp-340h] BYREF
  __int16 v45; // [rsp+80h] [rbp-320h]
  _BYTE *v46; // [rsp+90h] [rbp-310h] BYREF
  __int64 v47; // [rsp+98h] [rbp-308h]
  _BYTE v48[16]; // [rsp+A0h] [rbp-300h] BYREF
  __int16 v49; // [rsp+B0h] [rbp-2F0h]
  unsigned int *v50; // [rsp+D0h] [rbp-2D0h] BYREF
  int v51; // [rsp+D8h] [rbp-2C8h]
  _BYTE v52[40]; // [rsp+E0h] [rbp-2C0h] BYREF
  __int64 v53; // [rsp+108h] [rbp-298h]
  __int64 v54; // [rsp+110h] [rbp-290h]
  __int64 v55; // [rsp+120h] [rbp-280h]
  __int64 v56; // [rsp+128h] [rbp-278h]
  void *v57; // [rsp+150h] [rbp-250h]
  _BYTE *v58; // [rsp+160h] [rbp-240h] BYREF
  __int64 v59; // [rsp+168h] [rbp-238h]
  _BYTE v60[560]; // [rsp+170h] [rbp-230h] BYREF

  v39[1] = (__int64)&v58;
  v1 = a1[2];
  v2 = *a1;
  v58 = v60;
  v59 = 0x2000000000LL;
  v42 = 0;
  v43 = 0;
  v39[0] = (__int64)&v40;
  v3 = *(_QWORD *)(v1 + 16);
  v40 = 0;
  v41 = 0;
  sub_31CBFB0(v39, v2, v3);
  v4 = v59;
  if ( (_DWORD)v59 )
  {
    while ( 1 )
    {
      v5 = v4--;
      v6 = (__int64 *)&v58[16 * v5 - 16];
      v7 = *v6;
      v8 = v6[1];
      LODWORD(v59) = v4;
      v9 = *(_QWORD *)(v7 + 24);
      v10 = *(_BYTE *)v9;
      if ( *(_BYTE *)v9 <= 0x1Cu )
        break;
      if ( (unsigned __int8)(v10 - 78) <= 1u )
      {
        sub_23D0AB0((__int64)&v50, *(_QWORD *)(v7 + 24), 0, 0, 0);
        v45 = 257;
        v11 = (__int64 *)sub_BD5C60(v9);
        v12 = sub_BCE3C0(v11, 0);
        if ( v12 != *(_QWORD *)(v8 + 8) )
        {
          if ( *(_BYTE *)v8 > 0x15u )
          {
            v49 = 257;
            v8 = sub_B52190(v8, v12, (__int64)&v46, 0, 0);
            (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v56 + 16LL))(
              v56,
              v8,
              v44,
              v53,
              v54);
            v31 = v50;
            v32 = &v50[4 * v51];
            if ( v50 != v32 )
            {
              do
              {
                v33 = *((_QWORD *)v31 + 1);
                v34 = *v31;
                v31 += 4;
                sub_B99FD0(v8, v34, v33);
              }
              while ( v32 != v31 );
            }
          }
          else
          {
            v13 = *(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v55 + 144LL);
            if ( v13 == sub_B32D70 )
              v8 = sub_ADB060(v8, v12);
            else
              v8 = v13(v55, v8, v12);
            if ( *(_BYTE *)v8 > 0x1Cu )
            {
              (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v56 + 16LL))(
                v56,
                v8,
                v44,
                v53,
                v54);
              v14 = v50;
              v15 = &v50[4 * v51];
              if ( v50 != v15 )
              {
                do
                {
                  v16 = *((_QWORD *)v14 + 1);
                  v17 = *v14;
                  v14 += 4;
                  sub_B99FD0(v8, v17, v16);
                }
                while ( v15 != v14 );
              }
            }
          }
        }
        nullsub_61();
        v57 = &unk_49DA100;
        nullsub_63();
        if ( v50 != (unsigned int *)v52 )
          _libc_free((unsigned __int64)v50);
        sub_31CBFB0(v39, *(_QWORD *)(v7 + 24), v8);
        v4 = v59;
        goto LABEL_14;
      }
      if ( v10 == 63 )
      {
        sub_23D0AB0((__int64)&v50, *(_QWORD *)(v7 + 24), 0, 0, 0);
        v37 = *(_QWORD *)(v9 + 72);
        v20 = sub_B4DE30(v9);
        v21 = v37;
        v45 = 257;
        v22 = *(_DWORD *)(v9 + 4);
        v46 = v48;
        v23 = v22 & 0x7FFFFFF;
        v47 = 0x600000000LL;
        v24 = !v20 ? 0 : 3;
        v25 = (_QWORD *)(v9 + 32 * (1 - v23));
        v26 = (-32 * (1 - v23)) >> 5;
        if ( (unsigned __int64)(-32 * (1 - v23)) > 0xC0 )
        {
          v35 = v37;
          v36 = !v20 ? 0 : 3;
          v38 = (-32 * (1 - v23)) >> 5;
          sub_C8D5F0((__int64)&v46, v48, v38, 8u, v26, v24);
          v29 = v46;
          v28 = v47;
          LODWORD(v26) = v38;
          LODWORD(v24) = v36;
          v21 = v35;
          v27 = &v46[8 * (unsigned int)v47];
        }
        else
        {
          v27 = v48;
          v28 = 0;
          v29 = v48;
        }
        if ( (_QWORD *)v9 != v25 )
        {
          do
          {
            if ( v27 )
              *v27 = *v25;
            v25 += 4;
            ++v27;
          }
          while ( (_QWORD *)v9 != v25 );
          v29 = v46;
          v28 = v47;
        }
        LODWORD(v47) = v28 + v26;
        v30 = sub_921130(&v50, v21, v8, (_BYTE **)v29, (unsigned int)(v28 + v26), (__int64)v44, v24);
        if ( v46 != v48 )
          _libc_free((unsigned __int64)v46);
        sub_31CBFB0(v39, v9, v30);
        nullsub_61();
        v57 = &unk_49DA100;
        nullsub_63();
        if ( v50 != (unsigned int *)v52 )
          _libc_free((unsigned __int64)v50);
        goto LABEL_27;
      }
      if ( v10 != 62 )
        break;
LABEL_14:
      if ( !v4 )
        goto LABEL_15;
    }
    if ( *(_QWORD *)v7 )
    {
      v18 = *(_QWORD *)(v7 + 8);
      **(_QWORD **)(v7 + 16) = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = *(_QWORD *)(v7 + 16);
    }
    *(_QWORD *)v7 = v8;
    if ( v8 )
    {
      v19 = *(_QWORD *)(v8 + 16);
      *(_QWORD *)(v7 + 8) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = v7 + 8;
      *(_QWORD *)(v7 + 16) = v8 + 16;
      *(_QWORD *)(v8 + 16) = v7;
    }
LABEL_27:
    v4 = v59;
    goto LABEL_14;
  }
LABEL_15:
  sub_C7D6A0(v41, 8LL * (unsigned int)v43, 8);
  if ( v58 != v60 )
    _libc_free((unsigned __int64)v58);
}
