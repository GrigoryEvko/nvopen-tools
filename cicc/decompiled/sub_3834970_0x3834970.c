// Function: sub_3834970
// Address: 0x3834970
//
void __fastcall sub_3834970(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  __int128 v13; // rax
  _QWORD *v14; // r15
  const __m128i *v15; // r9
  __int128 *v16; // r13
  __int64 v17; // r8
  unsigned __int16 v18; // cx
  __int64 *v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // r13
  __int64 v22; // rax
  _QWORD *v23; // r11
  const __m128i *v24; // rdx
  unsigned __int16 v25; // cx
  __int64 v26; // r8
  unsigned int v27; // r15d
  __int128 v28; // rax
  __int64 v29; // r9
  unsigned __int64 v30; // r15
  __int64 v31; // rdx
  __int128 v32; // [rsp-30h] [rbp-E0h]
  __int64 v33; // [rsp+0h] [rbp-B0h]
  __int64 v34; // [rsp+0h] [rbp-B0h]
  unsigned __int16 v35; // [rsp+8h] [rbp-A8h]
  unsigned __int16 v36; // [rsp+8h] [rbp-A8h]
  _QWORD *v37; // [rsp+8h] [rbp-A8h]
  __int128 v38; // [rsp+10h] [rbp-A0h]
  const __m128i *v39; // [rsp+10h] [rbp-A0h]
  __int128 v40; // [rsp+10h] [rbp-A0h]
  const __m128i *v41; // [rsp+20h] [rbp-90h]
  _QWORD *v42; // [rsp+20h] [rbp-90h]
  __int64 v43; // [rsp+20h] [rbp-90h]
  __int64 v44; // [rsp+30h] [rbp-80h]
  __int64 v45; // [rsp+30h] [rbp-80h]
  unsigned __int64 v46; // [rsp+40h] [rbp-70h] BYREF
  __int64 v47; // [rsp+48h] [rbp-68h]
  __m128i v48; // [rsp+50h] [rbp-60h] BYREF
  __int64 v49; // [rsp+60h] [rbp-50h] BYREF
  __int64 v50; // [rsp+68h] [rbp-48h]
  unsigned __int64 v51; // [rsp+70h] [rbp-40h]
  __int64 v52; // [rsp+78h] [rbp-38h]

  v4 = a3;
  v5 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v48.m128i_i32[2] = 0;
  v46 = 0;
  LODWORD(v47) = 0;
  v6 = *(_QWORD *)(v5 + 8);
  v48.m128i_i64[0] = 0;
  if ( !(unsigned __int8)sub_3761870(a1, a2, *(_WORD *)v5, v6, 1) )
  {
    switch ( *(_DWORD *)(a2 + 24) )
    {
      case 3:
        sub_381DEB0(a1, a2, (unsigned int *)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 4:
        sub_381E300((__int64)a1, a2, (unsigned int *)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0xB:
        sub_381E9D0(a1, a2, (__int64)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0x33:
        sub_384A050(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0x34:
        sub_384A3C0(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0x35:
        sub_3846070(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0x36:
        sub_3846040(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0x37:
        sub_3849100(a1, a2, (unsigned int)v4, &v46, &v48);
        goto LABEL_4;
      case 0x38:
      case 0x39:
        sub_381B8F0(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x3A:
        sub_38217A0(a1, a2, (__int64)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0x3B:
        sub_38244A0(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x3C:
        sub_38265B0(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x3D:
        sub_3826030(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x3E:
        sub_3826A60(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x44:
      case 0x45:
        sub_381CDF0((__int64)a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x46:
      case 0x47:
        sub_381D0D0((__int64)a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x48:
      case 0x49:
        sub_381DA10((__int64)a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x4A:
      case 0x4B:
        sub_381DC70((__int64)a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x4C:
      case 0x4E:
        sub_3823DF0((__int64)a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x4D:
      case 0x4F:
        sub_381D320((__int64)a1, a2, &v46, (__int64)&v48);
        goto LABEL_4;
      case 0x50:
      case 0x51:
        sub_382BC40(a1, a2, (__int64)&v46, (unsigned __int8 **)&v48, v8, v9, a4);
        goto LABEL_4;
      case 0x52:
      case 0x53:
      case 0x54:
      case 0x55:
        v11 = sub_3466E90((_DWORD *)*a1, a2, a1[1], v7, v8, v9);
        goto LABEL_9;
      case 0x56:
      case 0x57:
        v11 = sub_3468870(*a1, a2, a1[1], v7, v8, v9);
        goto LABEL_9;
      case 0x58:
      case 0x59:
      case 0x5A:
      case 0x5B:
        sub_3821CB0(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x5C:
      case 0x5D:
      case 0x5E:
      case 0x5F:
        sub_3823CB0(a1, a2, (__int64)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0x87:
      case 0x88:
      case 0x89:
      case 0x8A:
      case 0x113:
      case 0x114:
      case 0x115:
      case 0x116:
        sub_3820160(a1, a2, (__int64)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0x8D:
      case 0x8E:
      case 0xE2:
      case 0xE3:
        sub_3834440(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x9E:
        sub_3846160(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0xAE:
      case 0xAF:
      case 0xB0:
      case 0xB1:
        v11 = (__int64)sub_345B810(*a1, a2, a1[1]);
        goto LABEL_9;
      case 0xB2:
      case 0xB3:
        v11 = sub_345AE90((_DWORD *)*a1, a2, a1[1], v7, v8, v9, a4);
        goto LABEL_9;
      case 0xB4:
      case 0xB5:
      case 0xB6:
      case 0xB7:
        sub_381AA30(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0xB8:
      case 0xB9:
        v11 = sub_34680A0((_DWORD *)*a1, a2, (_QWORD *)a1[1], a4);
        goto LABEL_9;
      case 0xBA:
      case 0xBB:
      case 0xBC:
        sub_3821620((__int64)a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0xBD:
        sub_381ED50(a1, a2, (unsigned int *)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0xBE:
      case 0xBF:
      case 0xC0:
        sub_3825320(a1, a2, (__int64)&v46, &v48, a4);
        goto LABEL_4;
      case 0xC1:
      case 0xC2:
        sub_3827040(a1, a2, (__int64)&v46, (__int64)&v48, a4, v8, v9);
        goto LABEL_4;
      case 0xC3:
      case 0xC4:
        sub_3827100(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0xC5:
        sub_381E790((__int64)a1, a2, (unsigned int *)&v46, (unsigned int *)&v48, a4);
        goto LABEL_4;
      case 0xC6:
      case 0xCB:
        sub_381FAD0(a1, a2, (unsigned int *)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0xC7:
      case 0xCC:
        sub_381F530(a1, a2, (unsigned int *)&v46, (unsigned __int8 **)&v48, a4);
        goto LABEL_4;
      case 0xC8:
        sub_381F970((__int64)a1, a2, (unsigned int *)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0xC9:
        sub_381E680((__int64)a1, a2, (unsigned int *)&v46, (unsigned int *)&v48, a4);
        goto LABEL_4;
      case 0xCA:
        sub_381E8A0((__int64)a1, a2, (unsigned int *)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0xCD:
        sub_3849200(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0xCF:
        sub_3849C90(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0xD0:
        sub_381A8B0(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0xD5:
        sub_382F8A0(a1, a2, (__int64)&v46, (unsigned int *)&v48, a4);
        goto LABEL_4;
      case 0xD6:
        sub_382FDF0(a1, a2, (__int64)&v46, v48.m128i_i64, a4);
        goto LABEL_4;
      case 0xD7:
        sub_382F520(a1, a2, (__int64)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0xD8:
        sub_38262A0(a1, a2, (__int64)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0xDE:
        sub_3825B60(a1, a2, (__int64)&v46, (unsigned int *)&v48, a4);
        goto LABEL_4;
      case 0xE4:
      case 0xE5:
        v11 = (__int64)sub_346B4B0((_BYTE *)*a1, a2, a1[1]);
        goto LABEL_9;
      case 0xE7:
        sub_381FF10(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0xEA:
        sub_384A7E0(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0x12A:
        sub_3820610(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x13D:
        sub_3846C20(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0x146:
      case 0x147:
        sub_3821B70(a1, a2, (__int64)&v46, (__int64)&v48);
        goto LABEL_4;
      case 0x14F:
        sub_384A5D0(a1, a2, &v46, &v48);
        goto LABEL_4;
      case 0x152:
        sub_3826F10((__int64)a1, a2, a4, (__int64)&v46, (__int64)&v48, v8, v9);
        goto LABEL_4;
      case 0x154:
      case 0x156:
      case 0x157:
      case 0x158:
      case 0x159:
      case 0x15A:
      case 0x15B:
      case 0x15C:
      case 0x15D:
      case 0x15E:
      case 0x15F:
      case 0x160:
      case 0x161:
        sub_3817A70((__int64)&v49, a1, a2);
        sub_375BC20(a1, v49, v50, (__int64)&v46, (__int64)&v48, a4);
        sub_3760E70((__int64)a1, a2, 1, v51, v52);
        goto LABEL_4;
      case 0x155:
        *(_QWORD *)&v13 = sub_33E5110(
                            (__int64 *)a1[1],
                            **(unsigned __int16 **)(a2 + 48),
                            *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                            1,
                            0);
        v14 = (_QWORD *)a1[1];
        v15 = *(const __m128i **)(a2 + 112);
        v16 = *(__int128 **)(a2 + 40);
        v17 = *(_QWORD *)(a2 + 104);
        v49 = *(_QWORD *)(a2 + 80);
        v18 = *(_WORD *)(a2 + 96);
        if ( v49 )
        {
          v35 = *(_WORD *)(a2 + 96);
          v38 = v13;
          v44 = v17;
          v41 = v15;
          sub_3813810(&v49);
          v18 = v35;
          v13 = v38;
          v17 = v44;
          v15 = v41;
        }
        LODWORD(v50) = *(_DWORD *)(a2 + 72);
        v19 = sub_33E6F00(
                v14,
                340,
                (__int64)&v49,
                v18,
                v17,
                v15,
                v13,
                *((__int64 *)&v13 + 1),
                *v16,
                *(__int128 *)((char *)v16 + 40),
                v16[5],
                *(__int128 *)((char *)v16 + 120));
        v45 = v20;
        v21 = (unsigned __int64)v19;
        sub_9C6650(&v49);
        v22 = *(_QWORD *)(a2 + 48);
        v23 = (_QWORD *)a1[1];
        v24 = *(const __m128i **)(a2 + 40);
        v25 = *(_WORD *)(v22 + 16);
        v26 = *(_QWORD *)(v22 + 24);
        v49 = *(_QWORD *)(a2 + 80);
        if ( v49 )
        {
          v33 = v26;
          v36 = v25;
          v39 = v24;
          v42 = v23;
          sub_3813810(&v49);
          v26 = v33;
          v25 = v36;
          v24 = v39;
          v23 = v42;
        }
        v34 = v26;
        v27 = v25;
        LODWORD(v50) = *(_DWORD *)(a2 + 72);
        v37 = v23;
        v40 = (__int128)_mm_loadu_si128(v24 + 5);
        *(_QWORD *)&v28 = sub_33ED040(v23, 0x11u);
        *((_QWORD *)&v32 + 1) = v45;
        *(_QWORD *)&v32 = v21;
        v30 = sub_340F900(v37, 0xD0u, (__int64)&v49, v27, v34, v29, v32, v40, v28);
        v43 = v31;
        sub_9C6650(&v49);
        sub_375BC20(a1, v21, v45, (__int64)&v46, (__int64)&v48, (__m128i)v40);
        sub_3760E70((__int64)a1, a2, 1, v30, v43);
        sub_3760E70((__int64)a1, a2, 2, v21, 1);
        goto LABEL_4;
      case 0x175:
        sub_38276A0(a1, a2, (__int64)&v46, (__int64)&v48, a4);
        goto LABEL_4;
      case 0x17E:
      case 0x17F:
      case 0x180:
      case 0x181:
      case 0x182:
      case 0x183:
      case 0x184:
      case 0x185:
      case 0x186:
        v11 = (__int64)sub_346A7D0(*a1, a2, (_QWORD *)a1[1], a4);
LABEL_9:
        sub_375BC20(a1, v11, v12, (__int64)&v46, (__int64)&v48, a4);
LABEL_4:
        if ( v46 )
          sub_375FFB0((__int64)a1, a2, v4, v46, v47, v10, v48.m128i_u64[0], v48.m128i_i64[1]);
        break;
      default:
        sub_C64ED0("Do not know how to expand the result of this operator!", 1u);
    }
  }
}
