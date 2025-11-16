// Function: sub_2924690
// Address: 0x2924690
//
__int64 __fastcall sub_2924690(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  unsigned __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned int v11; // r15d
  __int64 v12; // rdx
  __int64 *v13; // rdx
  __int64 v14; // r12
  unsigned int v15; // eax
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  __int64 *v19; // r14
  _QWORD *v20; // rdi
  __int64 v21; // r11
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 *v25; // rdx
  unsigned __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 *v29; // rax
  _QWORD *v30; // rsi
  _QWORD *v31; // rdx
  _QWORD *v32; // rax
  __int64 v33; // rcx
  int v34; // eax
  __int64 *v35; // rax
  char v36; // [rsp+7h] [rbp-179h]
  __int64 v37; // [rsp+8h] [rbp-178h]
  __int64 v38; // [rsp+10h] [rbp-170h]
  __int64 v39; // [rsp+28h] [rbp-158h]
  __int64 v40; // [rsp+30h] [rbp-150h]
  __int64 v41; // [rsp+38h] [rbp-148h]
  __int64 v42; // [rsp+40h] [rbp-140h]
  __int64 *v43; // [rsp+48h] [rbp-138h]
  __int64 v44; // [rsp+48h] [rbp-138h]
  __int64 v45; // [rsp+58h] [rbp-128h] BYREF
  __int64 v46[4]; // [rsp+60h] [rbp-120h] BYREF
  _QWORD v47[4]; // [rsp+80h] [rbp-100h] BYREF
  __int16 v48; // [rsp+A0h] [rbp-E0h]
  __int64 *v49; // [rsp+B0h] [rbp-D0h] BYREF
  _BYTE *v50; // [rsp+B8h] [rbp-C8h]
  __int64 v51; // [rsp+C0h] [rbp-C0h]
  _BYTE v52[16]; // [rsp+C8h] [rbp-B8h] BYREF
  _QWORD *v53; // [rsp+D8h] [rbp-A8h]
  __int64 v54; // [rsp+E0h] [rbp-A0h]
  _QWORD v55[6]; // [rsp+E8h] [rbp-98h] BYREF
  char v56; // [rsp+118h] [rbp-68h]
  __int64 v57; // [rsp+120h] [rbp-60h]
  __int64 v58; // [rsp+128h] [rbp-58h]
  __int64 v59; // [rsp+130h] [rbp-50h]
  __int64 v60; // [rsp+138h] [rbp-48h]
  __int64 v61; // [rsp+140h] [rbp-40h]
  __int64 v62; // [rsp+148h] [rbp-38h]

  v6 = 0;
  sub_2914720(a1, a2, a3, a4, a5, a6);
  v11 = *(_DWORD *)(a1 + 8);
  if ( v11 )
  {
    while ( 2 )
    {
      v12 = v11--;
      v13 = *(__int64 **)(*(_QWORD *)a1 + 8 * v12 - 8);
      *(_DWORD *)(a1 + 8) = v11;
      *(_QWORD *)(a1 + 176) = v13;
      v14 = v13[3];
      switch ( *(_BYTE *)v14 )
      {
        case 0x1E:
        case 0x1F:
        case 0x20:
        case 0x21:
        case 0x22:
        case 0x23:
        case 0x24:
        case 0x25:
        case 0x26:
        case 0x27:
        case 0x28:
        case 0x29:
        case 0x2A:
        case 0x2B:
        case 0x2C:
        case 0x2D:
        case 0x2E:
        case 0x2F:
        case 0x30:
        case 0x31:
        case 0x32:
        case 0x33:
        case 0x34:
        case 0x35:
        case 0x36:
        case 0x37:
        case 0x38:
        case 0x39:
        case 0x3A:
        case 0x3B:
        case 0x3C:
        case 0x40:
        case 0x41:
        case 0x42:
        case 0x43:
        case 0x44:
        case 0x45:
        case 0x46:
        case 0x47:
        case 0x48:
        case 0x49:
        case 0x4A:
        case 0x4B:
        case 0x4C:
        case 0x4D:
        case 0x50:
        case 0x51:
        case 0x52:
        case 0x53:
        case 0x55:
        case 0x57:
        case 0x58:
        case 0x59:
        case 0x5A:
        case 0x5B:
        case 0x5C:
        case 0x5D:
        case 0x5E:
        case 0x5F:
        case 0x60:
          goto LABEL_5;
        case 0x3D:
          v34 = sub_291E9F0(a1, v13[3]);
          v11 = *(_DWORD *)(a1 + 8);
          v6 |= v34;
          if ( !v11 )
            return v6;
          continue;
        case 0x3E:
          v43 = v13;
          if ( sub_B46500((unsigned __int8 *)v13[3]) )
            goto LABEL_5;
          if ( (*(_BYTE *)(v14 + 2) & 1) != 0 )
            goto LABEL_5;
          if ( *(_QWORD *)(v14 - 32) != *v43 )
            goto LABEL_5;
          v45 = *(_QWORD *)(v14 - 64);
          v17 = *(unsigned __int8 *)(*(_QWORD *)(v45 + 8) + 8LL);
          if ( (unsigned __int8)v17 <= 3u )
            goto LABEL_5;
          if ( (_BYTE)v17 == 5 )
            goto LABEL_5;
          if ( (unsigned __int8)v17 <= 0x14u )
          {
            v18 = 1463376;
            if ( _bittest64(&v18, v17) )
              goto LABEL_5;
          }
          v19 = *(__int64 **)(a1 + 192);
          v44 = *(_QWORD *)(a1 + 184);
          v36 = sub_2912520(v14, 0);
          sub_B91FC0(v46, v14);
          v40 = v46[2];
          v20 = (_QWORD *)v19[9];
          v42 = v46[0];
          v21 = **(_QWORD **)(a1 + 176);
          v37 = *(_QWORD *)(v45 + 8);
          v51 = 0x400000000LL;
          v38 = v21;
          v41 = v46[1];
          v39 = v46[3];
          v49 = v19;
          v50 = v52;
          v22 = sub_BCB2D0(v20);
          v55[0] = sub_ACD640(v22, 0, 0);
          v54 = 0x400000001LL;
          v55[4] = v38;
          v55[5] = v37;
          v56 = v36;
          v57 = v44;
          v53 = v55;
          sub_D5F1F0((__int64)v19, v14);
          v62 = v14;
          v60 = v40;
          v58 = v42;
          v59 = v41;
          v61 = v39;
          v47[0] = sub_BD5D20(v45);
          v47[2] = ".fca";
          v48 = 773;
          v47[1] = v23;
          sub_2923BA0(&v49, *(_QWORD *)(v45 + 8), &v45, (__int64)v47, v24, (__int64)&v49);
          v29 = *(__int64 **)(a1 + 176);
          if ( *(_BYTE *)*v29 > 0x1Cu )
            sub_2914720(a1, *v29, v25, v26, v27, v28);
          if ( *(_BYTE *)(a1 + 108) )
          {
            v30 = *(_QWORD **)(a1 + 88);
            v31 = &v30[*(unsigned int *)(a1 + 100)];
            v32 = v30;
            if ( v30 == v31 )
              goto LABEL_25;
            while ( v14 != *v32 )
            {
              if ( v31 == ++v32 )
                goto LABEL_25;
            }
            v33 = (unsigned int)(*(_DWORD *)(a1 + 100) - 1);
            *(_DWORD *)(a1 + 100) = v33;
            *v32 = v30[v33];
            ++*(_QWORD *)(a1 + 80);
          }
          else
          {
            v35 = sub_C8CA60(a1 + 80, v14);
            if ( v35 )
            {
              *v35 = -2;
              ++*(_DWORD *)(a1 + 104);
              ++*(_QWORD *)(a1 + 80);
            }
          }
LABEL_25:
          sub_AE94E0(v14);
          sub_B43D60((_QWORD *)v14);
          if ( v53 != v55 )
            _libc_free((unsigned __int64)v53);
          if ( v50 != v52 )
            _libc_free((unsigned __int64)v50);
          v11 = *(_DWORD *)(a1 + 8);
          v6 = 1;
          if ( !v11 )
            return v6;
          continue;
        case 0x3F:
          v15 = sub_291EF30(a1, (unsigned __int8 *)v13[3]);
          if ( !(_BYTE)v15 )
          {
            v15 = sub_29208D0(a1, v14);
            if ( !(_BYTE)v15 )
              goto LABEL_4;
          }
          v11 = *(_DWORD *)(a1 + 8);
          v6 = v15;
          if ( !v11 )
            return v6;
          continue;
        case 0x4E:
        case 0x4F:
        case 0x54:
        case 0x56:
LABEL_4:
          sub_2914720(a1, v14, v13, v8, v9, v10);
          v11 = *(_DWORD *)(a1 + 8);
LABEL_5:
          if ( !v11 )
            return v6;
          continue;
        default:
          BUG();
      }
    }
  }
  return v6;
}
