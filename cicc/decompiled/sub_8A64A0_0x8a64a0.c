// Function: sub_8A64A0
// Address: 0x8a64a0
//
_BOOL8 __fastcall sub_8A64A0(__int64 a1, __int64 a2, int a3, __int64 **a4)
{
  __int64 v7; // rdx
  __int64 v8; // rsi
  char v9; // dl
  __int64 v10; // r14
  __int64 *v11; // rax
  __int64 v12; // r8
  __int64 v13; // r13
  __int64 *v14; // rsi
  char v15; // al
  __int64 v16; // rcx
  __int64 v17; // r15
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _BOOL4 v24; // r13d
  __int64 *v26; // rdi
  __m128i *v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rcx
  _UNKNOWN *__ptr32 *v30; // r8
  __int64 *v31; // r9
  char v32; // al
  _BOOL4 v33; // r12d
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 *v37; // r9
  __int64 v38; // rax
  __int64 v39; // [rsp+8h] [rbp-C8h]
  __int16 v41; // [rsp+1Eh] [rbp-B2h]
  int v42; // [rsp+20h] [rbp-B0h]
  int v43; // [rsp+24h] [rbp-ACh]
  __int64 v44; // [rsp+28h] [rbp-A8h]
  __m128i *v45; // [rsp+28h] [rbp-A8h]
  int v46; // [rsp+34h] [rbp-9Ch] BYREF
  __int64 v47; // [rsp+38h] [rbp-98h] BYREF
  __m128i v48[9]; // [rsp+40h] [rbp-90h] BYREF

  v7 = sub_892920(a1);
  switch ( *(_BYTE *)(v7 + 80) )
  {
    case 4:
    case 5:
      v8 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 80LL);
      break;
    case 6:
      v8 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 32LL);
      break;
    case 9:
    case 0xA:
      v8 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 56LL);
      break;
    case 0x13:
    case 0x14:
    case 0x15:
    case 0x16:
      v8 = *(_QWORD *)(v7 + 88);
      break;
    default:
      v8 = 0;
      break;
  }
  v9 = *(_BYTE *)(a1 + 80);
  switch ( v9 )
  {
    case 4:
    case 5:
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      if ( v9 != 7 )
        goto LABEL_43;
      goto LABEL_7;
    case 6:
    case 10:
      goto LABEL_53;
    case 9:
      v10 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      goto LABEL_7;
    case 19:
    case 20:
    case 21:
    case 22:
      v10 = *(_QWORD *)(a1 + 88);
      if ( v9 != 19 )
        goto LABEL_5;
      v12 = *(_QWORD *)(v10 + 176);
      v13 = **(_QWORD **)(v10 + 256);
      v39 = **(_QWORD **)(v8 + 256);
      if ( a4 )
        goto LABEL_12;
      goto LABEL_9;
    default:
      v10 = 0;
LABEL_5:
      if ( v9 == 9 || v9 == 7 )
      {
LABEL_7:
        v11 = *(__int64 **)(a1 + 88);
      }
      else
      {
LABEL_43:
        if ( v9 != 21 )
LABEL_53:
          BUG();
        v11 = *(__int64 **)(*(_QWORD *)(a1 + 88) + 192LL);
      }
      v12 = *v11;
      v13 = **(_QWORD **)(v10 + 232);
      v39 = **(_QWORD **)(v8 + 232);
      if ( a4 )
      {
LABEL_12:
        v43 = 0;
        v14 = *a4;
      }
      else
      {
LABEL_9:
        v47 = 0;
        v14 = 0;
        a4 = (__int64 **)&v47;
        v43 = 1;
      }
      v44 = v12;
      *a4 = (__int64 *)sub_8A3C00(v13, (__int64)v14, 0, (__int64 *)&dword_4F077C8);
      v15 = *(_BYTE *)(a2 + 80);
      if ( v15 == 3 )
      {
        v17 = **(_QWORD **)(*(_QWORD *)(a2 + 88) + 168LL);
      }
      else
      {
        v16 = *(_QWORD *)(a2 + 88);
        if ( (unsigned __int8)(v15 - 4) <= 1u )
        {
          v17 = *(_QWORD *)(*(_QWORD *)(v16 + 168) + 168LL);
        }
        else if ( v15 == 7 )
        {
          v17 = **(_QWORD **)(v16 + 216);
        }
        else
        {
          v17 = *(_QWORD *)(v16 + 240);
        }
      }
      v18 = *(_BYTE *)(v44 + 80);
      v19 = *(_QWORD *)(v44 + 88);
      if ( v18 == 3 )
      {
        v45 = **(__m128i ***)(v19 + 168);
      }
      else if ( (unsigned __int8)(v18 - 4) <= 1u )
      {
        v45 = *(__m128i **)(*(_QWORD *)(v19 + 168) + 168LL);
      }
      else if ( v18 == 7 )
      {
        v45 = **(__m128i ***)(v19 + 216);
      }
      else
      {
        v45 = *(__m128i **)(v19 + 240);
      }
      if ( !(unsigned int)sub_8B4AF0(v17, v45, a4, v13, 512)
        || qword_4F074B0 && (unsigned int)sub_893F30(*a4, (__int64)v45, v20, v21, v22, v23) )
      {
        goto LABEL_24;
      }
      v42 = dword_4F07508[0];
      v41 = dword_4F07508[1];
      sub_865900(a1);
      v26 = *a4;
      v27 = (__m128i *)a1;
      if ( !(unsigned int)sub_8B59E0(*a4, a1, v13, 0, 0) )
        goto LABEL_34;
      if ( !a3 )
      {
        v27 = (__m128i *)*a4;
        v26 = (__int64 *)a1;
        if ( !(unsigned int)sub_8A00C0(a1, *a4, 0) )
          goto LABEL_34;
      }
      v46 = 0;
      sub_892150(v48);
      v26 = (__int64 *)a1;
      v27 = sub_8A55D0(a1, v45, v39, 0, (__int64)*a4, v13, (__int64 *)(a1 + 48), 1024, &v46, v48);
      if ( *(_BYTE *)(a1 + 80) == 19 )
      {
        v38 = *(_QWORD *)(v10 + 200);
        if ( v38 )
          v10 = *(_QWORD *)(v38 + 88);
      }
      v32 = *(_BYTE *)(v10 + 160);
      v28 = 2 * (unsigned int)((v32 & 6) != 0);
      v29 = (2 * ((v32 & 6) != 0)) | 0x20u;
      if ( (v32 & 0x10) != 0 )
        v28 = (unsigned int)v29;
      if ( v46 )
      {
LABEL_34:
        sub_864110((__int64)v26, (__int64)v27, v28, v29, (__int64)v30, v31);
        dword_4F07508[0] = v42;
        LOWORD(dword_4F07508[1]) = v41;
LABEL_24:
        v24 = 0;
        goto LABEL_25;
      }
      BYTE1(v28) |= 4u;
      v33 = sub_89AB40(v17, (__int64)v27, v28, v29, v30);
      v24 = v33;
      sub_864110(v17, (__int64)v27, v34, v35, v36, v37);
      dword_4F07508[0] = v42;
      LOWORD(dword_4F07508[1]) = v41;
      if ( !v33 | v43 )
LABEL_25:
        sub_725130(*a4);
      return v24;
  }
}
