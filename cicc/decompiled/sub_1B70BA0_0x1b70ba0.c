// Function: sub_1B70BA0
// Address: 0x1b70ba0
//
__int64 __fastcall sub_1B70BA0(__int64 a1, unsigned int a2, __int64 *a3, _BYTE *a4, double a5, double a6, double a7)
{
  __int64 v8; // rdx
  _QWORD *v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 ***v16; // r15
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rbx
  int v19; // r13d
  __int64 **v20; // rax
  __int64 *v21; // r14
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 **v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 result; // rax
  unsigned int v28; // esi
  int v29; // eax
  char v30; // si
  _QWORD *v31; // rax
  __int64 v32; // r15
  unsigned int v33; // eax
  __int64 v34; // r9
  unsigned __int64 v35; // r13
  __int64 v36; // rax
  __int64 ***v37; // r15
  __int64 **v38; // rax
  unsigned int v39; // ebx
  _QWORD *v40; // rax
  __int64 **v41; // rax
  __int64 v42; // r15
  _QWORD *v43; // rax
  __int64 v44; // rax
  __int64 *v45; // r14
  _QWORD *v46; // rax
  __int64 v47; // rax
  __int64 ***v48; // r14
  __int64 **v49; // rax
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // [rsp+20h] [rbp-60h]
  __int64 v54; // [rsp+28h] [rbp-58h]
  __int64 v55; // [rsp+28h] [rbp-58h]
  _QWORD *v56; // [rsp+28h] [rbp-58h]
  _QWORD *v58; // [rsp+38h] [rbp-48h]
  _QWORD *v59; // [rsp+38h] [rbp-48h]
  __int64 v60; // [rsp+38h] [rbp-48h]
  __int64 *v61; // [rsp+40h] [rbp-40h] BYREF
  _BYTE v62[56]; // [rsp+48h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v8 + 16) )
LABEL_16:
    BUG();
  if ( *(_DWORD *)(v8 + 36) == 137
    && *(_BYTE *)(*(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))) + 16LL) > 0x10u )
  {
    return 0;
  }
  v11 = (_QWORD *)*a3;
  v12 = (__int64)a3;
  v13 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v12 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v36 = *(_QWORD *)(v12 + 32);
        v12 = *(_QWORD *)(v12 + 24);
        v13 *= v36;
        continue;
      case 1:
        v14 = 16;
        goto LABEL_7;
      case 2:
        v14 = 32;
        goto LABEL_7;
      case 3:
      case 9:
        v14 = 64;
        goto LABEL_7;
      case 4:
        v14 = 80;
        goto LABEL_7;
      case 5:
      case 6:
        v14 = 128;
        goto LABEL_7;
      case 7:
        v54 = v13;
        v28 = 0;
        v58 = (_QWORD *)*a3;
        goto LABEL_18;
      case 0xB:
        v14 = *(_DWORD *)(v12 + 8) >> 8;
        goto LABEL_7;
      case 0xD:
        v55 = v13;
        v59 = (_QWORD *)*a3;
        v31 = (_QWORD *)sub_15A9930((__int64)a4, v12);
        v8 = *(_QWORD *)(a1 - 24);
        v11 = v59;
        v13 = v55;
        v30 = *(_BYTE *)(v8 + 16);
        v14 = 8LL * *v31;
        goto LABEL_19;
      case 0xE:
        v32 = *(_QWORD *)(v12 + 24);
        v53 = v13;
        v56 = (_QWORD *)*a3;
        v60 = *(_QWORD *)(v12 + 32);
        v33 = sub_15A9FE0((__int64)a4, v32);
        v13 = v53;
        v11 = v56;
        v34 = 1;
        v35 = v33;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v32 + 8) )
          {
            case 0:
              v52 = *(_QWORD *)(v32 + 32);
              v32 = *(_QWORD *)(v32 + 24);
              v34 *= v52;
              continue;
            case 1:
              v51 = 16;
              goto LABEL_33;
            case 2:
              v51 = 32;
              goto LABEL_33;
            case 3:
              JUMPOUT(0x1B70F55);
            case 4:
              v51 = 80;
              goto LABEL_33;
            case 5:
            case 6:
              v51 = 128;
LABEL_33:
              v8 = *(_QWORD *)(a1 - 24);
              v30 = *(_BYTE *)(v8 + 16);
              v14 = 8 * v35 * v60 * ((v35 + ((unsigned __int64)(v51 * v34 + 7) >> 3) - 1) / v35);
              break;
          }
          goto LABEL_19;
        }
      case 0xF:
        v54 = v13;
        v58 = (_QWORD *)*a3;
        v28 = *(_DWORD *)(v12 + 8) >> 8;
LABEL_18:
        v29 = sub_15A9520((__int64)a4, v28);
        v8 = *(_QWORD *)(a1 - 24);
        v11 = v58;
        v13 = v54;
        v14 = (unsigned int)(8 * v29);
        v30 = *(_BYTE *)(v8 + 16);
LABEL_19:
        if ( v30 )
          goto LABEL_16;
LABEL_7:
        v15 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
        v16 = *(__int64 ****)(a1 + 24 * (1 - v15));
        if ( *(_DWORD *)(v8 + 36) == 137 )
        {
          v17 = (unsigned __int64)(v14 * v13) >> 3;
          v18 = v17;
          if ( v17 != 1 )
          {
            v19 = 1;
            v20 = (__int64 **)sub_1644900(v11, 8 * (int)v17);
            v16 = (__int64 ***)sub_15A45D0(v16, v20);
            v21 = (__int64 *)v16;
            do
            {
              while ( 1 )
              {
                v24 = *v16;
                if ( v18 >= (unsigned int)(2 * v19) )
                  break;
                ++v19;
                v22 = sub_15A0680((__int64)v24, 8, 0);
                v23 = sub_15A2D50((__int64 *)v16, v22, 0, 0, a5, a6, a7);
                v16 = (__int64 ***)sub_15A2D10(v21, v23, a5, a6, a7);
                if ( v18 == v19 )
                  return sub_1B6E4B0((__int64)v16, (__int64)a3, a4, a5, a6, a7);
              }
              v25 = sub_15A0680((__int64)v24, (unsigned int)(8 * v19), 0);
              v26 = sub_15A2D50((__int64 *)v16, v25, 0, 0, a5, a6, a7);
              v16 = (__int64 ***)sub_15A2D10((__int64 *)v16, v26, a5, a6, a7);
              v19 *= 2;
            }
            while ( v18 != v19 );
          }
          return sub_1B6E4B0((__int64)v16, (__int64)a3, a4, a5, a6, a7);
        }
        v37 = (__int64 ***)sub_1649C60(*(_QWORD *)(a1 + 24 * (1 - v15)));
        v38 = *v37;
        if ( *((_BYTE *)*v37 + 8) == 16 )
          v38 = (__int64 **)*v38[2];
        v39 = *((_DWORD *)v38 + 2);
        v40 = (_QWORD *)sub_16498A0((__int64)v37);
        v39 >>= 8;
        v41 = (__int64 **)sub_16471D0(v40, v39);
        v42 = sub_15A4510(v37, v41, 0);
        v43 = (_QWORD *)sub_16498A0(v42);
        v44 = sub_1643360(v43);
        v45 = (__int64 *)sub_159C470(v44, a2, 0);
        v46 = (_QWORD *)sub_16498A0(v42);
        v47 = sub_1643330(v46);
        v61 = v45;
        v62[4] = 0;
        v48 = (__int64 ***)sub_15A2E80(v47, v42, &v61, 1u, 0, (__int64)v62, 0);
        v49 = (__int64 **)sub_1646BA0(a3, v39);
        v50 = sub_15A4510(v48, v49, 0);
        result = sub_14D8290(v50, (__int64)a3, a4);
        break;
    }
    return result;
  }
}
