// Function: sub_94DCB0
// Address: 0x94dcb0
//
__int64 __fastcall sub_94DCB0(__int64 a1, __int64 a2, int a3, __int64 a4, char a5, char a6)
{
  __int64 v6; // rax
  unsigned int v9; // ebx
  __int64 v10; // r15
  __int64 v11; // r12
  __m128i *v12; // r15
  __m128i *v13; // rax
  __int64 *v14; // rdi
  __m128i *v15; // r12
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // r14
  __int64 v19; // rax
  __int64 v20; // r12
  unsigned __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rax
  int v24; // ebx
  __int64 v25; // rax
  char v26; // al
  __int16 v27; // cx
  __int64 v28; // rax
  int v29; // r9d
  __int64 v30; // r12
  unsigned int *v31; // r14
  unsigned int *v32; // rbx
  __int64 v33; // rdx
  __int64 v34; // rsi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v40; // [rsp+10h] [rbp-C0h]
  __m128i *v41; // [rsp+18h] [rbp-B8h]
  __int64 *v42; // [rsp+20h] [rbp-B0h]
  char v43; // [rsp+30h] [rbp-A0h]
  int v44; // [rsp+30h] [rbp-A0h]
  __int16 v45; // [rsp+36h] [rbp-9Ah]
  __int64 v46; // [rsp+38h] [rbp-98h]
  __m128i *v47; // [rsp+38h] [rbp-98h]
  unsigned int i; // [rsp+38h] [rbp-98h]
  unsigned int v49; // [rsp+44h] [rbp-8Ch] BYREF
  __int64 v50; // [rsp+48h] [rbp-88h] BYREF
  _QWORD v51[4]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE v52[32]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v53; // [rsp+90h] [rbp-40h]

  v6 = (unsigned int)(a3 - 678);
  if ( (unsigned int)v6 > 0x1D )
  {
    v36 = (unsigned int)(a3 - 708);
    if ( (unsigned int)v36 > 0x17 )
    {
      v37 = (unsigned int)(a3 - 732);
      if ( (unsigned int)v37 > 0xC )
        sub_91B980("unexpected WMMA intrinsic!", 0);
      v43 = 0;
      v9 = dword_3F147A0[v37];
    }
    else
    {
      v43 = 0;
      v9 = dword_3F147E0[v36];
    }
  }
  else
  {
    v43 = 1;
    v9 = dword_3F14840[v6];
  }
  v10 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL) + 16LL);
  v42 = *(__int64 **)(*(_QWORD *)(a4 + 72) + 16LL);
  v11 = *(_QWORD *)(*(_QWORD *)(v10 + 16) + 16LL);
  v46 = *(_QWORD *)(v10 + 16);
  sub_9480A0(v11, 1u, "unexpected 'rowcol' operand", "'rowcol' operand can be 0 or 1 only", (_DWORD *)(a4 + 36));
  v41 = sub_92F410(a2, (__int64)v42);
  v12 = sub_92F410(a2, v10);
  v47 = sub_92F410(a2, v46);
  v13 = sub_92F410(a2, v11);
  v14 = *(__int64 **)(a2 + 32);
  v15 = v13;
  v50 = v12->m128i_i64[1];
  v16 = sub_90A810(v14, v9, (__int64)&v50, 1u);
  v51[0] = v12;
  v17 = 0;
  v51[2] = v15;
  v51[1] = v47;
  v53 = 257;
  if ( v16 )
    v17 = *(_QWORD *)(v16 + 24);
  v40 = sub_921880((unsigned int **)(a2 + 48), v17, v16, (int)v51, 3, (__int64)v52, 0);
  if ( !v43 )
  {
    if ( v9 <= 0x22CF )
    {
      if ( v9 > 0x22B2 )
      {
        switch ( v9 )
        {
          case 0x22B3u:
          case 0x22B4u:
          case 0x22B5u:
          case 0x22B6u:
          case 0x22CFu:
            v44 = 2;
            goto LABEL_9;
          case 0x22B7u:
          case 0x22BFu:
          case 0x22C7u:
            goto LABEL_18;
          case 0x22BBu:
          case 0x22BCu:
          case 0x22C5u:
          case 0x22C6u:
            goto LABEL_8;
          case 0x22BDu:
          case 0x22BEu:
          case 0x22C3u:
          case 0x22C4u:
          case 0x22CBu:
          case 0x22CCu:
          case 0x22CDu:
          case 0x22CEu:
            goto LABEL_22;
          default:
            goto LABEL_25;
        }
      }
      if ( v9 <= 0x2055 )
      {
LABEL_22:
        v44 = 1;
        goto LABEL_9;
      }
      v44 = 2;
      if ( v9 == 8278 )
        goto LABEL_9;
    }
LABEL_25:
    sub_91B980("unexpected imma_ld intrinsic!", 0);
  }
  if ( a6 != 1 || a5 )
LABEL_18:
    v44 = 8;
  else
LABEL_8:
    v44 = 4;
LABEL_9:
  for ( i = 0; i != v44; ++i )
  {
    v18 = v40;
    if ( v44 != 1 )
    {
      v53 = 257;
      v49 = i;
      v18 = sub_94D3D0((unsigned int **)(a2 + 48), v40, (__int64)&v49, 1, (__int64)v52);
    }
    v19 = *(_QWORD *)(a2 + 32);
    v53 = 257;
    v20 = v19 + 8;
    v21 = sub_8D46C0(*v42);
    v23 = sub_91A390(v20, v21, 0, v22);
    v24 = sub_94B2B0((unsigned int **)(a2 + 48), v23, (__int64)v41, i, (__int64)v52);
    v25 = sub_AA4E30(*(_QWORD *)(a2 + 96));
    v26 = sub_AE5020(v25, *(_QWORD *)(v18 + 8));
    HIBYTE(v27) = HIBYTE(v45);
    v53 = 257;
    LOBYTE(v27) = v26;
    v45 = v27;
    v28 = sub_BD2C40(80, unk_3F10A10);
    v30 = v28;
    if ( v28 )
      sub_B4D3C0(v28, v18, v24, 0, (unsigned __int8)v45, v29, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v30,
      v52,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112));
    v31 = *(unsigned int **)(a2 + 48);
    v32 = &v31[4 * *(unsigned int *)(a2 + 56)];
    while ( v32 != v31 )
    {
      v33 = *((_QWORD *)v31 + 1);
      v34 = *v31;
      v31 += 4;
      sub_B99FD0(v30, v34, v33);
    }
  }
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  return a1;
}
