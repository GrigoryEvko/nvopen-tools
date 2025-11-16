// Function: sub_89FDA0
// Address: 0x89fda0
//
__int64 __fastcall sub_89FDA0(__int64 a1, __int64 a2, __int64 a3, int a4, int a5)
{
  char v7; // al
  __int64 **v8; // rcx
  __int64 *v9; // rax
  __int64 v10; // r15
  char v11; // al
  __int64 v12; // rbx
  __int8 *v13; // rax
  unsigned int v14; // eax
  __int64 *v15; // rsi
  unsigned int v16; // r15d
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 *v21; // r9
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 *v25; // r9
  _DWORD *v27; // r12
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 *v30; // [rsp-10h] [rbp-100h]
  int v33; // [rsp+14h] [rbp-DCh] BYREF
  __int64 v34; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v35[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v36[2]; // [rsp+30h] [rbp-C0h] BYREF
  const __m128i *v37; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v38; // [rsp+48h] [rbp-A8h]
  __int64 v39; // [rsp+50h] [rbp-A0h]
  __int64 v40[9]; // [rsp+60h] [rbp-90h] BYREF
  int v41; // [rsp+A8h] [rbp-48h]

  v7 = *(_BYTE *)(a1 + 80);
  v33 = 0;
  v8 = *(__int64 ***)(a1 + 88);
  if ( v7 == 20 )
  {
    v10 = *v8[41];
  }
  else
  {
    if ( v7 == 21 )
      v9 = v8[29];
    else
      v9 = v8[4];
    v10 = *v9;
  }
  v39 = 0;
  v34 = *(_QWORD *)dword_4F07508;
  v38 = 0;
  v37 = (const __m128i *)sub_823970(0);
  sub_865900(a1);
  v35[0] = 0;
  v35[1] = 0;
  sub_892150(v40);
  if ( a5 )
  {
    v11 = *(_BYTE *)(a1 + 80);
    if ( v11 == 20 )
    {
      v28 = *(_QWORD *)(a1 + 88);
    }
    else
    {
      if ( v11 != 10 )
        goto LABEL_8;
      v28 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
    }
    v29 = *(_QWORD *)(v28 + 176);
    v41 = 0;
    sub_88FF90((__int64)v40, **(__int64 ***)(*(_QWORD *)(v29 + 152) + 168LL));
    if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 && *(_BYTE *)(a1 + 80) == 20 )
      sub_89F970(*(_QWORD *)(a1 + 64), (__int64)&v37);
  }
LABEL_8:
  v12 = v39;
  if ( v39 == v38 )
    sub_738390(&v37);
  v13 = &v37->m128i_i8[24 * v12];
  if ( v13 )
  {
    v13[16] &= 0xF0u;
    *(_QWORD *)v13 = v10;
    *((_QWORD *)v13 + 1) = a3;
  }
  v39 = v12 + 1;
  v14 = sub_6F0CB0(a2, (unsigned __int64)&v37, (__m128i *)v35, 0, v40, &v33, 0);
  v15 = v30;
  v16 = v14;
  if ( v14 )
  {
    v16 = 1;
  }
  else if ( v35[0] )
  {
    if ( a4 || v33 && !(_DWORD)qword_4F077B4 )
    {
      v36[0] = 0;
      v36[1] = 0;
      sub_686D90(0xC3Fu, (FILE *)(a2 + 28), a3, v36);
      sub_67E390(v36, v35, 0);
      v15 = v35;
      v27 = sub_67D9D0(0xBE4u, &v34);
      sub_67E370((__int64)v27, (const __m128i *)v35);
      sub_685910((__int64)v27, (FILE *)v35);
    }
    else
    {
      v16 = 0;
      sub_67E3D0(v35);
    }
  }
  v17 = v40[0];
  sub_8921C0(v40[0]);
  sub_864110(v17, (__int64)v15, v18, v19, v20, v21);
  sub_823A00((__int64)v37, 24 * v38, v22, v23, v24, v25);
  return v16;
}
