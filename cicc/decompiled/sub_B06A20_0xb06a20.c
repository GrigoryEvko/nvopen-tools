// Function: sub_B06A20
// Address: 0xb06a20
//
__int64 __fastcall sub_B06A20(
        __int64 *a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        unsigned int a10,
        __int64 a11,
        __int64 a12,
        int a13,
        unsigned int a14,
        __int64 a15,
        int a16,
        __int64 a17,
        __int64 a18,
        __int64 a19,
        unsigned __int64 a20,
        __int64 a21,
        __int64 a22,
        __int64 a23,
        __int64 a24,
        __int64 a25)
{
  __int64 v28; // rbx
  unsigned int v29; // esi
  int v30; // r10d
  __int64 v31; // r8
  __int64 *v32; // rdx
  unsigned int v33; // r11d
  __int64 *v34; // rax
  __int64 v35; // rcx
  __int64 *v36; // rbx
  __int64 result; // rax
  __int64 v38; // rdx
  bool v39; // dl
  __int64 v40; // r13
  __int64 v41; // r14
  __int64 i; // rdi
  __int64 v43; // rax
  __int64 v44; // r8
  int v45; // ecx
  int v46; // eax
  __int64 v49; // [rsp+28h] [rbp-B8h] BYREF
  _QWORD v50[22]; // [rsp+30h] [rbp-B0h] BYREF

  if ( !(unsigned __int8)sub_B6F8F0() )
    return 0;
  v28 = *a1;
  v49 = a2;
  v29 = *(_DWORD *)(v28 + 1648);
  if ( v29 )
  {
    v30 = 1;
    v31 = *(_QWORD *)(v28 + 1632);
    v32 = 0;
    v33 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v34 = (__int64 *)(v31 + 16LL * v33);
    v35 = *v34;
    if ( a2 == *v34 )
    {
LABEL_4:
      v36 = v34 + 1;
      goto LABEL_5;
    }
    while ( v35 != -4096 )
    {
      if ( !v32 && v35 == -8192 )
        v32 = v34;
      v33 = (v29 - 1) & (v30 + v33);
      v34 = (__int64 *)(v31 + 16LL * v33);
      v35 = *v34;
      if ( a2 == *v34 )
        goto LABEL_4;
      ++v30;
    }
    if ( !v32 )
      v32 = v34;
    v50[0] = v32;
    v46 = *(_DWORD *)(v28 + 1640);
    ++*(_QWORD *)(v28 + 1624);
    v45 = v46 + 1;
    if ( 4 * (v46 + 1) < 3 * v29 )
    {
      v44 = a2;
      if ( v29 - *(_DWORD *)(v28 + 1644) - v45 > v29 >> 3 )
        goto LABEL_28;
      goto LABEL_27;
    }
  }
  else
  {
    v50[0] = 0;
    ++*(_QWORD *)(v28 + 1624);
  }
  v29 *= 2;
LABEL_27:
  sub_B00180(v28 + 1624, v29);
  sub_AF6CB0(v28 + 1624, &v49, v50);
  v44 = v49;
  v32 = (__int64 *)v50[0];
  v45 = *(_DWORD *)(v28 + 1640) + 1;
LABEL_28:
  *(_DWORD *)(v28 + 1640) = v45;
  if ( *v32 != -4096 )
    --*(_DWORD *)(v28 + 1644);
  *v32 = v44;
  v36 = v32 + 1;
  v32[1] = 0;
LABEL_5:
  if ( !*v36 )
  {
    v50[0] = a17;
    result = sub_B065E0(
               a1,
               a3,
               a4,
               a5,
               a6,
               a7,
               a8,
               a9,
               a10,
               a11,
               a14,
               a15,
               a16,
               a17,
               a18,
               a19,
               a2,
               a20,
               a21,
               a22,
               a23,
               a24,
               a25,
               a12,
               a13,
               1u,
               1);
    *v36 = result;
    return result;
  }
  if ( (unsigned __int16)sub_AF18C0(*v36) != a3 )
    return 0;
  result = *v36;
  if ( (*(_BYTE *)(*v36 + 20) & 4) != 0 && (a14 & 4) == 0 )
  {
    v50[0] = a17;
    *(_DWORD *)(result + 44) = a16;
    v38 = v50[0];
    *(_WORD *)(result + 2) = a3;
    *(_DWORD *)(result + 16) = a6;
    *(_QWORD *)(result + 48) = v38;
    *(_DWORD *)(result + 20) = a14;
    *(_QWORD *)(result + 24) = a9;
    *(_DWORD *)(result + 4) = a10;
    *(_QWORD *)(result + 32) = a11;
    *(_DWORD *)(result + 40) = a13;
    v50[7] = a2;
    v50[1] = a7;
    v50[2] = a4;
    v50[3] = a8;
    v50[4] = a15;
    v50[5] = a18;
    v50[6] = a19;
    v50[8] = a20;
    v50[9] = a21;
    v50[10] = a22;
    v50[11] = a23;
    v50[12] = a24;
    v50[13] = a25;
    v50[14] = a12;
    result = *v36;
    v39 = (*(_BYTE *)(*v36 - 16) & 2) != 0;
    v40 = (*(_BYTE *)(*v36 - 16) & 2) != 0 ? *(unsigned int *)(result - 24) : (*(_WORD *)(result - 16) >> 6) & 0xFu;
    if ( (_DWORD)v40 )
    {
      v41 = 0;
      for ( i = *v36; ; v39 = (*(_BYTE *)(i - 16) & 2) != 0 )
      {
        v43 = v39 ? *(_QWORD *)(i - 32) : i - 16 - 8LL * ((*(_BYTE *)(i - 16) >> 2) & 0xF);
        if ( *(_QWORD *)(v43 + 8 * v41) != a5 )
        {
          sub_B97110(i, (unsigned int)v41, a5);
          i = *v36;
        }
        if ( v40 == ++v41 )
          break;
        a5 = v50[v41];
      }
      return i;
    }
  }
  return result;
}
