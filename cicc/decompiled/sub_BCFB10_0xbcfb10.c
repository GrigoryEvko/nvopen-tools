// Function: sub_BCFB10
// Address: 0xbcfb10
//
__int64 __fastcall sub_BCFB10(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        const void *a5,
        __int64 a6,
        void *a7,
        __int64 a8)
{
  _QWORD *v12; // r14
  char v13; // al
  __int64 v14; // rdi
  const void *v15; // r8
  int v17; // eax
  unsigned int v18; // esi
  _QWORD *v19; // r11
  int v20; // ecx
  _QWORD *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rsi
  unsigned __int64 v24; // r14
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rax
  const void *v30; // [rsp+20h] [rbp-80h]
  unsigned __int64 *v31; // [rsp+28h] [rbp-78h]
  _QWORD *v32; // [rsp+28h] [rbp-78h]
  _QWORD *v33; // [rsp+30h] [rbp-70h] BYREF
  _QWORD *v34; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v35[12]; // [rsp+40h] [rbp-60h] BYREF

  v12 = (_QWORD *)*a2;
  v35[0] = a3;
  v35[2] = a5;
  v35[1] = a4;
  v35[3] = a6;
  v35[4] = a7;
  v35[5] = a8;
  v13 = sub_BCCA50((__int64)(v12 + 375), (__int64)v35, &v33);
  v14 = (__int64)(v12 + 375);
  v15 = a5;
  if ( v13 )
  {
    *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
    *(_QWORD *)a1 = *v33;
    return a1;
  }
  v17 = *((_DWORD *)v12 + 754);
  v18 = *((_DWORD *)v12 + 756);
  v19 = v33;
  ++v12[375];
  v20 = v17 + 1;
  v34 = v19;
  if ( 4 * (v17 + 1) >= 3 * v18 )
  {
    v18 *= 2;
LABEL_14:
    sub_BCF880(v14, v18);
    sub_BCCA50(v14, (__int64)v35, &v34);
    v19 = v34;
    v15 = a5;
    v20 = *((_DWORD *)v12 + 754) + 1;
    goto LABEL_6;
  }
  if ( v18 - *((_DWORD *)v12 + 755) - v20 <= v18 >> 3 )
    goto LABEL_14;
LABEL_6:
  *((_DWORD *)v12 + 754) = v20;
  if ( *v19 != -4096 )
    --*((_DWORD *)v12 + 755);
  *v19 = 0;
  v21 = (_QWORD *)*a2;
  v22 = *(_QWORD *)(*a2 + 2640LL);
  v23 = 4 * (a8 + 2 * a6 + 12);
  v21[340] += v23;
  v24 = (v22 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v21[331] >= v23 + v24 && v22 )
  {
    v21[330] = v23 + v24;
  }
  else
  {
    v30 = v15;
    v32 = v19;
    v27 = sub_9D1E70((__int64)(v21 + 330), v23, v23, 3);
    v15 = v30;
    v19 = v32;
    v24 = v27;
  }
  v31 = v19;
  sub_BCBD60(v24, a2, a3, a4, v15, a6, a7, a8);
  *v31 = v24;
  sub_BCBE20(a1, v24, v25, v26);
  return a1;
}
