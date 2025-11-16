// Function: sub_375A8B0
// Address: 0x375a8b0
//
unsigned __int8 *__fastcall sub_375A8B0(__int64 a1, __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned int v4; // r15d
  unsigned __int16 *v6; // rdx
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rdx
  _QWORD *v12; // rdi
  unsigned __int16 v13; // cx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned __int16 *v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  bool v20; // di
  unsigned __int64 v21; // rsi
  int v22; // eax
  __int64 v23; // r9
  unsigned int v24; // edx
  __int64 v25; // r8
  __int64 v26; // rsi
  unsigned __int8 *v27; // r12
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // [rsp+8h] [rbp-88h]
  __int64 *v32; // [rsp+10h] [rbp-80h]
  __int64 v33; // [rsp+18h] [rbp-78h]
  unsigned int v35; // [rsp+20h] [rbp-70h]
  __int64 v36; // [rsp+20h] [rbp-70h]
  unsigned __int16 v37; // [rsp+20h] [rbp-70h]
  __int64 v38; // [rsp+28h] [rbp-68h]
  __int64 v39; // [rsp+28h] [rbp-68h]
  unsigned __int64 v40; // [rsp+30h] [rbp-60h] BYREF
  __int64 v41; // [rsp+38h] [rbp-58h]
  __int64 v42; // [rsp+40h] [rbp-50h] BYREF
  __int64 v43; // [rsp+48h] [rbp-48h]
  __int64 v44; // [rsp+50h] [rbp-40h]
  __int64 v45; // [rsp+58h] [rbp-38h]

  v33 = 16LL * a3;
  v6 = (unsigned __int16 *)(v33 + *(_QWORD *)(a2 + 48));
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  LOWORD(v42) = v7;
  v43 = v8;
  if ( (_WORD)v7 )
  {
    if ( (unsigned __int16)(v7 - 17) > 0xD3u )
    {
      LOWORD(v40) = v7;
      v41 = v8;
      goto LABEL_4;
    }
    LOWORD(v7) = word_4456580[v7 - 1];
    v10 = 0;
  }
  else
  {
    v39 = v8;
    if ( !sub_30070B0((__int64)&v42) )
    {
      v41 = v39;
      LOWORD(v40) = 0;
      goto LABEL_9;
    }
    LOWORD(v7) = sub_3009970((__int64)&v42, a2, v39, v15, v16);
  }
  LOWORD(v40) = v7;
  v41 = v10;
  if ( !(_WORD)v7 )
  {
LABEL_9:
    v44 = sub_3007260((__int64)&v40);
    LODWORD(v9) = v44;
    v45 = v11;
    goto LABEL_10;
  }
LABEL_4:
  if ( (_WORD)v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
    BUG();
  v9 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v7 - 16];
LABEL_10:
  v12 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 64LL);
  v38 = *(_QWORD *)(a1 + 8);
  v32 = *(__int64 **)(v38 + 64);
  switch ( (_DWORD)v9 )
  {
    case 1:
      v13 = 2;
      break;
    case 2:
      v13 = 3;
      break;
    case 4:
      v13 = 4;
      break;
    case 8:
      v13 = 5;
      break;
    case 0x10:
      v13 = 6;
      break;
    case 0x20:
      v13 = 7;
      break;
    case 0x40:
      v13 = 8;
      break;
    case 0x80:
      v13 = 9;
      break;
    default:
      v13 = sub_3007020(v12, v9);
      v31 = v14;
      v38 = *(_QWORD *)(a1 + 8);
      v32 = *(__int64 **)(v38 + 64);
      goto LABEL_23;
  }
  v31 = 0;
LABEL_23:
  v17 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + v33);
  LODWORD(v18) = *v17;
  v19 = *((_QWORD *)v17 + 1);
  LOWORD(v42) = v18;
  v43 = v19;
  if ( (_WORD)v18 )
  {
    v20 = (unsigned __int16)(v18 - 176) <= 0x34u;
    LODWORD(v21) = word_4456340[(int)v18 - 1];
    LOBYTE(v18) = v20;
  }
  else
  {
    v37 = v13;
    v30 = sub_3007240((__int64)&v42);
    v13 = v37;
    v21 = v30;
    v18 = HIDWORD(v30);
    v40 = v21;
    v20 = v18;
  }
  LODWORD(v42) = v21;
  BYTE4(v42) = v18;
  v35 = v13;
  if ( v20 )
  {
    LOWORD(v22) = sub_2D43AD0(v13, v21);
    v24 = v35;
    v25 = 0;
    if ( (_WORD)v22 )
      goto LABEL_27;
  }
  else
  {
    LOWORD(v22) = sub_2D43050(v13, v21);
    v24 = v35;
    v25 = 0;
    if ( (_WORD)v22 )
      goto LABEL_27;
  }
  v22 = sub_3009450(v32, v24, v31, v42, 0, v23);
  HIWORD(v4) = HIWORD(v22);
  v25 = v29;
LABEL_27:
  v26 = *(_QWORD *)(a2 + 80);
  LOWORD(v4) = v22;
  v42 = v26;
  if ( v26 )
  {
    v36 = v25;
    sub_B96E90((__int64)&v42, v26, 1);
    v25 = v36;
  }
  LODWORD(v43) = *(_DWORD *)(a2 + 72);
  v27 = sub_33FAF80(v38, 234, (__int64)&v42, v4, v25, v23, a4);
  if ( v42 )
    sub_B91220((__int64)&v42, v42);
  return v27;
}
