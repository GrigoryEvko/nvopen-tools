// Function: sub_20CC3C0
// Address: 0x20cc3c0
//
__int64 __fastcall sub_20CC3C0(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  unsigned int v10; // r13d
  __int64 v12; // rdi
  __int64 (*v13)(void); // rax
  int v15; // eax
  unsigned int v16; // r13d
  __int64 v17; // rax
  unsigned int v18; // r13d
  __int64 v19; // rdi
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 v22; // rbx
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rdi
  double v26; // xmm4_8
  double v27; // xmm5_8
  __int64 v28; // rsi
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // r14
  unsigned __int64 v32; // r15
  __int64 v33; // rax
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 *v35; // [rsp+18h] [rbp-48h] BYREF
  __int64 **v36; // [rsp+28h] [rbp-38h] BYREF

  v10 = 0;
  v12 = *(_QWORD *)(a1 + 160);
  v35 = a2;
  v13 = *(__int64 (**)(void))(*(_QWORD *)v12 + 680LL);
  if ( v13 != sub_1F3CB30 )
  {
    v15 = v13();
    if ( v15 == 1 )
    {
      v24 = sub_15F2050((__int64)v35);
      v25 = sub_1632FA0(v24);
      v28 = *(_QWORD *)*(v35 - 3);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v28 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v28 = *(_QWORD *)(v28 + 24);
            continue;
          case 1:
          case 2:
          case 3:
          case 4:
          case 5:
          case 6:
          case 9:
          case 0xB:
            goto LABEL_10;
          case 7:
            sub_15A9520(v25, 0);
            break;
          case 0xD:
            sub_15A9930(v25, v28);
            break;
          case 0xE:
            v29 = *(_QWORD *)(v28 + 24);
            sub_15A9FE0(v25, v29);
            sub_127FA20(v25, v29);
            break;
          case 0xF:
            sub_15A9520(v25, *(_DWORD *)(v28 + 8) >> 8);
            break;
        }
        break;
      }
LABEL_10:
      v36 = &v35;
      v10 = 1;
      sub_20C96A0(
        a1,
        v35,
        *(v35 - 6),
        (*((unsigned __int16 *)v35 + 9) >> 2) & 7,
        (__int64 (__fastcall *)(__int64, unsigned __int8 **, __int64))sub_20CD3C0,
        (__int64)&v36,
        a3,
        a4,
        a5,
        a6,
        v26,
        v27,
        a9,
        a10);
    }
    else if ( v15 == 3 )
    {
      v16 = *(_DWORD *)(*(_QWORD *)(a1 + 160) + 104LL);
      v17 = sub_15F2050((__int64)v35);
      v18 = v16 >> 3;
      v19 = sub_1632FA0(v17);
      v22 = 1;
      v23 = *(_QWORD *)*(v35 - 3);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v23 + 8) )
        {
          case 1:
            v30 = 16;
            goto LABEL_12;
          case 2:
            v30 = 32;
            goto LABEL_12;
          case 3:
          case 9:
            v30 = 64;
            goto LABEL_12;
          case 4:
            v30 = 80;
            goto LABEL_12;
          case 5:
          case 6:
            v30 = 128;
            goto LABEL_12;
          case 7:
            v30 = 8 * (unsigned int)sub_15A9520(v19, 0);
            goto LABEL_12;
          case 0xB:
            v30 = *(_DWORD *)(v23 + 8) >> 8;
            goto LABEL_12;
          case 0xD:
            v30 = 8LL * *(_QWORD *)sub_15A9930(v19, v23);
            goto LABEL_12;
          case 0xE:
            v31 = *(_QWORD *)(v23 + 32);
            v34 = *(_QWORD *)(v23 + 24);
            v32 = (unsigned int)sub_15A9FE0(v19, v34);
            v30 = 8 * v32 * v31 * ((v32 + ((unsigned __int64)(sub_127FA20(v19, v34) + 7) >> 3) - 1) / v32);
            goto LABEL_12;
          case 0xF:
            v30 = 8 * (unsigned int)sub_15A9520(v19, *(_DWORD *)(v23 + 8) >> 8);
LABEL_12:
            if ( v18 <= (unsigned int)((unsigned __int64)(v30 * v22 + 7) >> 3) )
            {
              v10 = 1;
              sub_20CAAD0(
                v35,
                (void (__fastcall *)(__int64, __int64 *, __int64, __int64, __int64, _QWORD, unsigned __int8 **, __int64 *))sub_20C9610,
                (__int64)sub_20CAC10,
                a3,
                a4,
                a5,
                a6,
                v20,
                v21,
                a9,
                a10);
            }
            else
            {
              v10 = 1;
              sub_20CBD50(a1, (__int64)v35, a3, a4, a5, a6, v20, v21, a9, a10);
            }
            return v10;
          case 0x10:
            v33 = *(_QWORD *)(v23 + 32);
            v23 = *(_QWORD *)(v23 + 24);
            v22 *= v33;
            continue;
          default:
            BUG();
        }
      }
    }
  }
  return v10;
}
