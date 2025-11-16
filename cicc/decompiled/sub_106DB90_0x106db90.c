// Function: sub_106DB90
// Address: 0x106db90
//
__int64 *__fastcall sub_106DB90(__int64 *a1, _DWORD *a2, __int64 a3)
{
  bool v5; // r14
  _BYTE *v6; // rax
  __int64 v7; // rdi
  _BYTE *v9; // rax
  _BYTE *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rbx
  _BYTE *v13; // rax
  __int64 v14; // rax
  _BYTE *v15; // rax
  void (*v16)(void); // rax
  _BYTE *v17; // r15
  __int64 v18; // rax
  _BYTE *v19; // rax
  _BYTE *v20; // rax
  _BYTE *v21; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v22[7]; // [rsp+8h] [rbp-38h] BYREF

  (*(void (__fastcall **)(_BYTE **))(*(_QWORD *)a2 + 40LL))(&v21);
  v5 = a2[2] == 1;
  switch ( (*(unsigned int (__fastcall **)(_BYTE *))(*(_QWORD *)v21 + 16LL))(v21) )
  {
    case 1u:
      v9 = v21;
      v21 = 0;
      v22[0] = v9;
      sub_108ACE0(a1, v22, a3);
      v7 = v22[0];
      if ( !v22[0] )
        goto LABEL_4;
      goto LABEL_3;
    case 2u:
      v10 = v21;
      v21 = 0;
      v11 = sub_22077B0(128);
      v12 = v11;
      if ( v11 )
      {
        *(_QWORD *)(v11 + 16) = 0;
        *(_QWORD *)(v11 + 8) = v11 + 24;
        *(_QWORD *)(v11 + 24) = v11 + 40;
        *(_WORD *)(v11 + 80) = 0;
        *(_QWORD *)(v11 + 88) = v11 + 104;
        *(_QWORD *)(v11 + 32) = 0;
        *(_BYTE *)(v11 + 40) = 0;
        *(_QWORD *)(v11 + 56) = 0;
        *(_QWORD *)(v11 + 64) = 0;
        *(_QWORD *)(v11 + 72) = 0;
        *(_QWORD *)(v11 + 96) = 0;
        *(_QWORD *)v11 = &unk_49E6078;
        *(_QWORD *)(v11 + 104) = a3;
        *(_DWORD *)(v11 + 112) = 1;
        *(_QWORD *)(v11 + 120) = v10;
      }
      else if ( v10 )
      {
        (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v10 + 8LL))(v10);
      }
      goto LABEL_11;
    case 3u:
      v13 = v21;
      v21 = 0;
      v22[0] = v13;
      v14 = sub_22077B0(224);
      v12 = v14;
      if ( v14 )
        sub_124C640(v14, v22, a3, v5);
      if ( v22[0] )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v22[0] + 8LL))(v22[0]);
      goto LABEL_11;
    case 4u:
      v15 = v21;
      v21 = 0;
      v22[0] = v15;
      sub_1253350(a1, v22, a3);
      if ( v22[0] )
      {
        v16 = *(void (**)(void))(*(_QWORD *)v22[0] + 8LL);
        if ( (char *)v16 == (char *)sub_106DB80 )
          j_j___libc_free_0(v22[0], 8);
        else
          v16();
      }
      goto LABEL_4;
    case 5u:
      v17 = v21;
      v21 = 0;
      v18 = sub_22077B0(2064);
      v12 = v18;
      if ( v18 )
      {
        *(_QWORD *)(v18 + 8) = v18 + 24;
        *(_QWORD *)(v18 + 24) = v18 + 40;
        *(_QWORD *)(v18 + 88) = v18 + 104;
        *(_WORD *)(v18 + 80) = 0;
        *(_QWORD *)(v18 + 16) = 0;
        *(_QWORD *)v18 = &unk_49E60C0;
        *(_QWORD *)(v18 + 32) = 0;
        *(_BYTE *)(v18 + 40) = 0;
        *(_QWORD *)(v18 + 56) = 0;
        *(_QWORD *)(v18 + 64) = 0;
        *(_QWORD *)(v18 + 72) = 0;
        *(_QWORD *)(v18 + 96) = 0;
        *(_QWORD *)(v18 + 104) = v17;
        *(_QWORD *)(v18 + 112) = 0;
        *(_QWORD *)(v18 + 120) = 0;
        *(_QWORD *)(v18 + 128) = 0;
        *(_DWORD *)(v18 + 136) = 0;
        *(_QWORD *)(v18 + 144) = 0;
        *(_QWORD *)(v18 + 152) = 0;
        *(_QWORD *)(v18 + 160) = 0;
        *(_QWORD *)(v18 + 168) = 0;
        *(_QWORD *)(v18 + 176) = 0;
        *(_QWORD *)(v18 + 184) = 0;
        *(_DWORD *)(v18 + 192) = 0;
        *(_QWORD *)(v18 + 200) = 0;
        *(_QWORD *)(v18 + 208) = 0;
        *(_QWORD *)(v18 + 216) = 0;
        *(_QWORD *)(v18 + 224) = 0;
        *(_QWORD *)(v18 + 232) = 0;
        *(_QWORD *)(v18 + 240) = 0;
        *(_DWORD *)(v18 + 248) = 0;
        *(_QWORD *)(v18 + 256) = v18 + 272;
        *(_QWORD *)(v18 + 264) = 0;
        sub_C0BFB0(v18 + 272, ((v17[8] & 1) != 0) + 2, 0);
        *(_QWORD *)(v12 + 320) = 0;
        *(_QWORD *)(v12 + 400) = v12 + 416;
        *(_QWORD *)(v12 + 328) = 0;
        *(_QWORD *)(v12 + 336) = 0;
        *(_QWORD *)(v12 + 344) = 0;
        *(_QWORD *)(v12 + 352) = 0;
        *(_QWORD *)(v12 + 360) = 0;
        *(_QWORD *)(v12 + 368) = 0;
        *(_QWORD *)(v12 + 376) = 0;
        *(_QWORD *)(v12 + 384) = 0;
        *(_QWORD *)(v12 + 392) = 0;
        *(_QWORD *)(v12 + 408) = 0x2000000000LL;
        *(_DWORD *)(v12 + 1984) = 0;
        *(_DWORD *)(v12 + 2020) = 0;
        *(_QWORD *)(v12 + 2024) = 0;
        *(_QWORD *)(v12 + 2032) = 0;
        *(_QWORD *)(v12 + 2040) = 0;
        *(_QWORD *)(v12 + 2048) = a3;
        *(_DWORD *)(v12 + 2056) = v5;
        *(_OWORD *)(v12 + 1952) = 0;
        *(_OWORD *)(v12 + 1968) = 0;
        *(_OWORD *)(v12 + 1988) = 0;
        *(_OWORD *)(v12 + 2004) = 0;
      }
      else if ( v17 )
      {
        (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v17 + 8LL))(v17);
      }
LABEL_11:
      *a1 = v12;
      goto LABEL_4;
    case 6u:
      v19 = v21;
      v21 = 0;
      v22[0] = v19;
      sub_1076C10(a1, v22, a3);
      v7 = v22[0];
      if ( !v22[0] )
        goto LABEL_4;
      goto LABEL_3;
    case 7u:
      v20 = v21;
      v21 = 0;
      v22[0] = v20;
      sub_107BE40(a1, v22, a3);
      v7 = v22[0];
      if ( !v22[0] )
        goto LABEL_4;
      goto LABEL_3;
    case 8u:
      v6 = v21;
      v21 = 0;
      v22[0] = v6;
      sub_1090060(a1, v22, a3);
      v7 = v22[0];
      if ( v22[0] )
LABEL_3:
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
LABEL_4:
      if ( v21 )
        (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v21 + 8LL))(v21);
      return a1;
    default:
      BUG();
  }
}
