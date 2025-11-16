// Function: sub_94ED50
// Address: 0x94ed50
//
__int64 __fastcall sub_94ED50(__int64 a1, __int64 a2, unsigned int a3, unsigned __int64 *a4, char a5)
{
  __int64 v7; // r14
  __int64 v9; // r10
  unsigned __int64 *v10; // r9
  __int64 v11; // rdi
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  __int64 *v14; // rdi
  __int64 v15; // r15
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v20; // rax
  __int64 *v21; // rdi
  __int64 v22; // rbx
  unsigned __int64 v23; // rsi
  __int64 v24; // rbx
  __int64 v25; // rax
  unsigned __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r15
  __int64 v30; // rax
  _BYTE *v31; // r11
  __int64 v32; // rdi
  __int64 (__fastcall *v33)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v34; // rax
  __int64 v35; // rbx
  __int64 v36; // rax
  unsigned __int8 v37; // al
  __int64 v38; // rax
  int v39; // r9d
  __int64 v40; // r15
  unsigned int *v41; // rbx
  unsigned int *v42; // r12
  __int64 v43; // rdx
  __int64 v44; // rsi
  __int64 v45; // rax
  unsigned int *v46; // rax
  unsigned int *v47; // r15
  __int64 v48; // rdx
  __int64 v49; // rsi
  __int64 v50; // rax
  _BYTE *v51; // [rsp+8h] [rbp-108h]
  unsigned int v52; // [rsp+8h] [rbp-108h]
  _BYTE *v53; // [rsp+8h] [rbp-108h]
  _BYTE *v54; // [rsp+8h] [rbp-108h]
  __int64 v55; // [rsp+10h] [rbp-100h]
  __int64 v56; // [rsp+18h] [rbp-F8h]
  int v57; // [rsp+18h] [rbp-F8h]
  __int64 v58; // [rsp+20h] [rbp-F0h]
  __int64 v59; // [rsp+20h] [rbp-F0h]
  unsigned int *v60; // [rsp+20h] [rbp-F0h]
  __int64 v61; // [rsp+28h] [rbp-E8h]
  __int64 v62; // [rsp+28h] [rbp-E8h]
  int v63; // [rsp+34h] [rbp-DCh] BYREF
  __int64 v64; // [rsp+38h] [rbp-D8h] BYREF
  _QWORD v65[2]; // [rsp+40h] [rbp-D0h] BYREF
  char v66[32]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v67; // [rsp+70h] [rbp-A0h]
  _QWORD v68[4]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v69; // [rsp+A0h] [rbp-70h]
  _BYTE v70[32]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v71; // [rsp+D0h] [rbp-40h]

  v7 = a2 + 48;
  v9 = *(_QWORD *)(a4[9] + 16);
  v10 = *(unsigned __int64 **)(v9 + 16);
  v11 = *(_QWORD *)(a2 + 32) + 8LL;
  v12 = *v10;
  if ( !a5 )
  {
    v61 = *(_QWORD *)(v9 + 16);
    v58 = *(_QWORD *)(a4[9] + 16);
    v13 = sub_91A390(v11, v12, 0, (__int64)a4);
    v14 = *(__int64 **)(a2 + 32);
    v64 = v13;
    v15 = sub_90A810(v14, a3, (__int64)&v64, 1u);
    v68[0] = sub_92F410(a2, v58);
    v16 = 0;
    v68[1] = sub_92F410(a2, v61);
    v71 = 257;
    if ( v15 )
      v16 = *(_QWORD *)(v15 + 24);
    v17 = sub_921880((unsigned int **)v7, v16, v15, (int)v68, 2, (__int64)v70, 0);
    *(_BYTE *)(a1 + 12) &= ~1u;
    *(_DWORD *)(a1 + 8) = 0;
    *(_DWORD *)(a1 + 16) = 0;
    *(_QWORD *)a1 = v17;
    return a1;
  }
  v62 = *(_QWORD *)(v9 + 16);
  v59 = *(_QWORD *)(a4[9] + 16);
  v56 = v10[2];
  v20 = sub_91A390(v11, v12, 0, (__int64)a4);
  v21 = *(__int64 **)(a2 + 32);
  v64 = v20;
  v22 = sub_90A810(v21, a3, (__int64)&v64, 1u);
  v65[0] = sub_92F410(a2, v59);
  v23 = 0;
  v65[1] = sub_92F410(a2, v62);
  v71 = 257;
  if ( v22 )
    v23 = *(_QWORD *)(v22 + 24);
  v24 = sub_921880((unsigned int **)v7, v23, v22, (int)v65, 2, (__int64)v70, 0);
  v71 = 257;
  LODWORD(v68[0]) = 0;
  v25 = sub_94D3D0((unsigned int **)v7, v24, (__int64)v68, 1, (__int64)v70);
  v26 = *a4;
  v55 = v25;
  v27 = *(_QWORD *)(a2 + 32);
  v69 = 257;
  v29 = sub_91A390(v27 + 8, v26, 0, v28);
  v63 = 1;
  v67 = 257;
  v30 = sub_94D3D0((unsigned int **)v7, v24, (__int64)&v63, 1, (__int64)v66);
  v31 = (_BYTE *)v30;
  if ( v29 == *(_QWORD *)(v30 + 8) )
  {
    v35 = v30;
    goto LABEL_15;
  }
  v32 = *(_QWORD *)(a2 + 128);
  v33 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v32 + 120LL);
  if ( v33 == sub_920130 )
  {
    if ( *v31 > 0x15u )
    {
LABEL_22:
      v53 = v31;
      v71 = 257;
      v45 = sub_BD2C40(72, unk_3F10A14);
      v35 = v45;
      if ( v45 )
        sub_B515B0(v45, v53, v29, v70, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
        *(_QWORD *)(a2 + 136),
        v35,
        v68,
        *(_QWORD *)(v7 + 56),
        *(_QWORD *)(v7 + 64));
      v46 = *(unsigned int **)(a2 + 48);
      v47 = v46;
      v60 = &v46[4 * *(unsigned int *)(a2 + 56)];
      if ( v46 != v60 )
      {
        do
        {
          v48 = *((_QWORD *)v47 + 1);
          v49 = *v47;
          v47 += 4;
          sub_B99FD0(v35, v49, v48);
        }
        while ( v60 != v47 );
      }
      goto LABEL_15;
    }
    v51 = v31;
    if ( (unsigned __int8)sub_AC4810(39) )
      v34 = sub_ADAB70(39, v51, v29, 0);
    else
      v34 = sub_AA93C0(39, v51, v29);
    v31 = v51;
    v35 = v34;
  }
  else
  {
    v54 = v31;
    v50 = v33(v32, 39u, v31, v29);
    v31 = v54;
    v35 = v50;
  }
  if ( !v35 )
    goto LABEL_22;
LABEL_15:
  v52 = (unsigned int)sub_92F410(a2, v56);
  v36 = sub_AA4E30(*(_QWORD *)(a2 + 96));
  v37 = sub_AE5020(v36, *(_QWORD *)(v35 + 8));
  v71 = 257;
  v57 = v37;
  v38 = sub_BD2C40(80, unk_3F10A10);
  v40 = v38;
  if ( v38 )
    sub_B4D3C0(v38, v35, v52, 0, v57, v39, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
    *(_QWORD *)(a2 + 136),
    v40,
    v70,
    *(_QWORD *)(v7 + 56),
    *(_QWORD *)(v7 + 64));
  v41 = *(unsigned int **)(a2 + 48);
  v42 = &v41[4 * *(unsigned int *)(a2 + 56)];
  while ( v42 != v41 )
  {
    v43 = *((_QWORD *)v41 + 1);
    v44 = *v41;
    v41 += 4;
    sub_B99FD0(v40, v44, v43);
  }
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = v55;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
