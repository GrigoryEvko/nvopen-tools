// Function: sub_2F739C0
// Address: 0x2f739c0
//
__int64 __fastcall sub_2F739C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rcx
  int v12; // r11d
  unsigned int i; // eax
  __int64 v14; // r8
  unsigned int v15; // eax
  __int64 v16; // rax
  bool v17; // zf
  __int64 **v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  void **v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 **v36; // rsi
  __int64 v37; // [rsp+0h] [rbp-440h] BYREF
  void **v38; // [rsp+8h] [rbp-438h]
  unsigned int v39; // [rsp+10h] [rbp-430h]
  unsigned int v40; // [rsp+14h] [rbp-42Ch]
  char v41; // [rsp+1Ch] [rbp-424h]
  _BYTE v42[16]; // [rsp+20h] [rbp-420h] BYREF
  _BYTE v43[8]; // [rsp+30h] [rbp-410h] BYREF
  unsigned __int64 v44; // [rsp+38h] [rbp-408h]
  int v45; // [rsp+44h] [rbp-3FCh]
  int v46; // [rsp+48h] [rbp-3F8h]
  char v47; // [rsp+4Ch] [rbp-3F4h]
  _BYTE v48[16]; // [rsp+50h] [rbp-3F0h] BYREF
  _QWORD v49[8]; // [rsp+60h] [rbp-3E0h] BYREF
  _BYTE v50[320]; // [rsp+A0h] [rbp-3A0h] BYREF
  __int64 v51; // [rsp+1E0h] [rbp-260h]
  __int64 v52; // [rsp+1E8h] [rbp-258h]
  __int64 v53; // [rsp+1F0h] [rbp-250h]
  int v54; // [rsp+1F8h] [rbp-248h]
  __int64 v55; // [rsp+200h] [rbp-240h]
  __int64 v56; // [rsp+208h] [rbp-238h]
  __int64 v57; // [rsp+210h] [rbp-230h]
  int v58; // [rsp+218h] [rbp-228h]
  __int64 v59; // [rsp+220h] [rbp-220h]
  __int64 v60; // [rsp+228h] [rbp-218h]
  __int64 v61; // [rsp+230h] [rbp-210h]
  int v62; // [rsp+238h] [rbp-208h]
  __int128 v63; // [rsp+240h] [rbp-200h]
  __int16 v64; // [rsp+250h] [rbp-1F0h]
  char v65; // [rsp+252h] [rbp-1EEh]
  char *v66; // [rsp+258h] [rbp-1E8h]
  __int64 v67; // [rsp+260h] [rbp-1E0h]
  char v68; // [rsp+268h] [rbp-1D8h] BYREF
  char *v69; // [rsp+2A8h] [rbp-198h]
  __int64 v70; // [rsp+2B0h] [rbp-190h]
  char v71; // [rsp+2B8h] [rbp-188h] BYREF
  __int64 v72; // [rsp+2F8h] [rbp-148h]
  char *v73; // [rsp+300h] [rbp-140h]
  __int64 v74; // [rsp+308h] [rbp-138h]
  int v75; // [rsp+310h] [rbp-130h]
  char v76; // [rsp+314h] [rbp-12Ch]
  char v77; // [rsp+318h] [rbp-128h] BYREF
  char *v78; // [rsp+358h] [rbp-E8h]
  __int64 v79; // [rsp+360h] [rbp-E0h]
  char v80; // [rsp+368h] [rbp-D8h] BYREF
  char *v81; // [rsp+3A8h] [rbp-98h]
  __int64 v82; // [rsp+3B0h] [rbp-90h]
  char v83; // [rsp+3B8h] [rbp-88h] BYREF
  __int64 v84; // [rsp+3D8h] [rbp-68h]
  __int64 v85; // [rsp+3E0h] [rbp-60h]
  __int64 v86; // [rsp+3E8h] [rbp-58h]
  __int64 v87; // [rsp+3F0h] [rbp-50h]
  __int64 v88; // [rsp+3F8h] [rbp-48h]
  __int64 v89; // [rsp+400h] [rbp-40h]
  __int64 v90; // [rsp+408h] [rbp-38h]
  int v91; // [rsp+410h] [rbp-30h]

  *(_QWORD *)(a3 + 344) &= 0xFFEuLL;
  v7 = sub_2EB2140(a4, (__int64 *)&unk_501EAD0, a3) + 8;
  v8 = sub_2EB2140(a4, &qword_50208B0, a3);
  v9 = *(unsigned int *)(a4 + 88);
  v10 = *(_QWORD *)(a4 + 72);
  v11 = v8 + 8;
  if ( !(_DWORD)v9 )
    goto LABEL_25;
  v12 = 1;
  for ( i = (v9 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&qword_5025C20 >> 9) ^ ((unsigned int)&qword_5025C20 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v9 - 1) & v15 )
  {
    v14 = v10 + 24LL * i;
    if ( *(__int64 **)v14 == &qword_5025C20 && a3 == *(_QWORD *)(v14 + 8) )
      break;
    if ( *(_QWORD *)v14 == -4096 && *(_QWORD *)(v14 + 8) == -4096 )
      goto LABEL_25;
    v15 = v12 + i;
    ++v12;
  }
  if ( v14 == v10 + 24 * v9 )
  {
LABEL_25:
    v16 = 0;
  }
  else
  {
    v16 = *(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL);
    if ( v16 )
      v16 += 8;
  }
  v49[6] = v16;
  v49[0] = off_4A2B718;
  memset(&v49[1], 0, 32);
  v49[5] = v7;
  v49[7] = v11;
  sub_2F5FEE0((__int64)v50);
  v69 = &v71;
  v64 = 0;
  v73 = &v77;
  v66 = &v68;
  v78 = &v80;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v65 = 0;
  v67 = 0x800000000LL;
  v70 = 0x800000000LL;
  v72 = 0;
  v74 = 8;
  v75 = 0;
  v76 = 1;
  v79 = 0x800000000LL;
  v81 = &v83;
  v82 = 0x800000000LL;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v63 = 0;
  v88 = 0;
  v17 = *(_BYTE *)(a3 + 341) == 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  if ( !v17 || !(unsigned __int8)sub_2F71140((__int64)v49, (_QWORD *)a3) )
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
    goto LABEL_13;
  }
  sub_2EAFFB0((__int64)&v37);
  if ( v45 == v46 )
  {
    if ( v41 )
    {
      v23 = v38;
      v36 = (__int64 **)&v38[v40];
      v20 = v40;
      v19 = (__int64 **)v38;
      if ( v38 != (void **)v36 )
      {
        while ( *v19 != &qword_4F82400 )
        {
          if ( v36 == ++v19 )
          {
LABEL_19:
            while ( *v23 != &unk_4F82408 )
            {
              if ( ++v23 == (void **)v19 )
                goto LABEL_26;
            }
            goto LABEL_20;
          }
        }
        goto LABEL_20;
      }
      goto LABEL_26;
    }
    if ( sub_C8CA60((__int64)&v37, (__int64)&qword_4F82400) )
      goto LABEL_20;
  }
  if ( !v41 )
  {
LABEL_28:
    sub_C8CC70((__int64)&v37, (__int64)&unk_4F82408, (__int64)v19, v20, v21, v22);
    goto LABEL_20;
  }
  v23 = v38;
  v20 = v40;
  v19 = (__int64 **)&v38[v40];
  if ( v19 != (__int64 **)v38 )
    goto LABEL_19;
LABEL_26:
  if ( v39 <= (unsigned int)v20 )
    goto LABEL_28;
  v20 = (unsigned int)(v20 + 1);
  v40 = v20;
  *v19 = (__int64 *)&unk_4F82408;
  ++v37;
LABEL_20:
  sub_2F62D70((__int64)&v37, (__int64)&unk_501EAD0, (__int64)v19, v20, v21, v22);
  sub_2F62D70((__int64)&v37, (__int64)&qword_5025C20, v24, v25, v26, v27);
  sub_2F62D70((__int64)&v37, (__int64)&qword_50208B0, v28, v29, v30, v31);
  sub_2F62D70((__int64)&v37, (__int64)qword_501FE48, v32, v33, v34, v35);
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v42, (__int64)&v37);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v48, (__int64)v43);
  if ( !v47 )
    _libc_free(v44);
  if ( !v41 )
    _libc_free((unsigned __int64)v38);
LABEL_13:
  sub_2F61430((__int64)v49);
  return a1;
}
