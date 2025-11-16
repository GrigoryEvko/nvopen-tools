// Function: sub_1D94D00
// Address: 0x1d94d00
//
__int64 __fastcall sub_1D94D00(__int64 a1, __int64 a2, __int64 a3, char a4, char a5, unsigned int a6)
{
  char v6; // r10
  __int64 result; // rax
  __int64 v13; // rdi
  __int64 v14; // rcx
  const void *v15; // rsi
  __int64 v16; // rdx
  __int64 (*v17)(); // rax
  char v18; // al
  __int64 v19; // rax
  const void *v20; // r11
  size_t v21; // r8
  __int64 v22; // rax
  unsigned __int64 v23; // r9
  const void *v24; // r11
  __int64 v25; // rax
  __int64 v26; // r9
  int v27; // edx
  unsigned __int64 v28; // r8
  _BYTE *v29; // rdi
  __int64 (*v30)(); // rax
  __int64 v31; // r8
  __int64 (*v32)(); // rax
  _BYTE *v33; // rdi
  _BYTE *v34; // rdi
  __int64 (*v35)(); // rax
  __int64 v36; // [rsp+8h] [rbp-1B8h]
  size_t v37; // [rsp+10h] [rbp-1B0h]
  char v38; // [rsp+10h] [rbp-1B0h]
  int v39; // [rsp+18h] [rbp-1A8h]
  char v40; // [rsp+18h] [rbp-1A8h]
  const void *v41; // [rsp+18h] [rbp-1A8h]
  char v42; // [rsp+20h] [rbp-1A0h]
  int v43; // [rsp+20h] [rbp-1A0h]
  const void *v44; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v45; // [rsp+20h] [rbp-1A0h]
  char v47; // [rsp+28h] [rbp-198h]
  unsigned __int64 v48; // [rsp+28h] [rbp-198h]
  unsigned __int8 v49; // [rsp+28h] [rbp-198h]
  unsigned __int8 v50; // [rsp+28h] [rbp-198h]
  _BYTE *v51; // [rsp+30h] [rbp-190h] BYREF
  __int64 v52; // [rsp+38h] [rbp-188h]
  _BYTE v53[160]; // [rsp+40h] [rbp-180h] BYREF
  _BYTE *v54; // [rsp+E0h] [rbp-E0h] BYREF
  __int64 v55; // [rsp+E8h] [rbp-D8h]
  _BYTE dest[208]; // [rsp+F0h] [rbp-D0h] BYREF

  v6 = a5;
  if ( *(_DWORD *)(a2 + 224) )
  {
    result = (*(_BYTE *)a2 & 0x10) != 0;
    if ( (*(_BYTE *)a2 & 0x10) == 0 )
      return result;
    v13 = *(_QWORD *)(a1 + 544);
    v14 = *(_QWORD *)(a2 + 216);
    v15 = *(const void **)a3;
    v16 = *(unsigned int *)(a3 + 8);
    v17 = *(__int64 (**)())(*(_QWORD *)v13 + 704LL);
    if ( v17 == sub_1D918D0 )
      return 0;
    v18 = ((__int64 (__fastcall *)(__int64, const void *, __int64, __int64))v17)(v13, v15, v16, v14);
    v6 = a5;
    if ( !v18 )
      return 0;
  }
  result = a6;
  if ( !(_BYTE)a6 )
  {
    result = 1;
    if ( *(_DWORD *)(a2 + 48) )
    {
      if ( !a4 )
        return a6;
      v19 = *(unsigned int *)(a3 + 8);
      v20 = *(const void **)a3;
      v51 = v53;
      v52 = 0x400000000LL;
      v22 = 40 * v19;
      v21 = v22;
      v23 = 0xCCCCCCCCCCCCCCCDLL * (v22 >> 3);
      if ( (unsigned __int64)v22 > 0xA0 )
      {
        v37 = v22;
        v40 = v6;
        v44 = v20;
        v48 = 0xCCCCCCCCCCCCCCCDLL * (v22 >> 3);
        sub_16CD150((__int64)&v51, v53, v48, 40, v22, v23);
        LODWORD(v23) = v48;
        v20 = v44;
        v6 = v40;
        v21 = v37;
        v34 = &v51[40 * (unsigned int)v52];
      }
      else
      {
        if ( !v22 )
          goto LABEL_12;
        v34 = v53;
      }
      v43 = v23;
      v47 = v6;
      memcpy(v34, v20, v21);
      LODWORD(v22) = v52;
      LODWORD(v23) = v43;
      v6 = v47;
LABEL_12:
      v24 = *(const void **)(a2 + 40);
      LODWORD(v52) = v23 + v22;
      v25 = *(unsigned int *)(a2 + 48);
      v54 = dest;
      v26 = 40 * v25;
      v27 = 40 * v25;
      v55 = 0x400000000LL;
      v28 = 0xCCCCCCCCCCCCCCCDLL * ((40 * v25) >> 3);
      if ( (unsigned __int64)(40 * v25) > 0xA0 )
      {
        v36 = 40 * v25;
        v38 = v6;
        v41 = v24;
        v45 = 0xCCCCCCCCCCCCCCCDLL * (v26 >> 3);
        sub_16CD150((__int64)&v54, dest, v45, 40, v28, v26);
        LODWORD(v28) = v45;
        v24 = v41;
        v6 = v38;
        v26 = v36;
        v33 = &v54[40 * (unsigned int)v55];
      }
      else
      {
        if ( !v26 )
        {
LABEL_14:
          LODWORD(v55) = v27 + v28;
          if ( v6 )
          {
            v31 = *(_QWORD *)(a1 + 544);
            v29 = v54;
            v32 = *(__int64 (**)())(*(_QWORD *)v31 + 624LL);
            if ( v32 == sub_1D918B0 )
              goto LABEL_16;
            if ( ((unsigned __int8 (__fastcall *)(__int64, _BYTE **))v32)(v31, &v54) )
              goto LABEL_23;
          }
          v29 = v54;
          v30 = *(__int64 (**)())(**(_QWORD **)(a1 + 544) + 624LL);
          if ( v30 == sub_1D918B0 )
          {
LABEL_16:
            if ( v29 != dest )
              _libc_free((unsigned __int64)v29);
            if ( v51 != v53 )
              _libc_free((unsigned __int64)v51);
            return a6;
          }
          if ( ((unsigned __int8 (__fastcall *)(_QWORD, _BYTE **))v30)(*(_QWORD *)(a1 + 544), &v51) )
          {
LABEL_23:
            v29 = v54;
            goto LABEL_16;
          }
          v29 = v54;
          v35 = *(__int64 (**)())(**(_QWORD **)(a1 + 544) + 704LL);
          if ( v35 == sub_1D918D0 )
            goto LABEL_16;
          result = ((__int64 (__fastcall *)(_QWORD, _BYTE *, _QWORD, _BYTE *, _QWORD))v35)(
                     *(_QWORD *)(a1 + 544),
                     v54,
                     (unsigned int)v55,
                     v51,
                     (unsigned int)v52);
          v29 = v54;
          if ( !(_BYTE)result )
            goto LABEL_16;
          if ( v54 != dest )
          {
            v49 = result;
            _libc_free((unsigned __int64)v54);
            result = v49;
          }
          if ( v51 != v53 )
          {
            v50 = result;
            _libc_free((unsigned __int64)v51);
            return v50;
          }
          return result;
        }
        v33 = dest;
      }
      v39 = v28;
      v42 = v6;
      memcpy(v33, v24, v26);
      v27 = v55;
      LODWORD(v28) = v39;
      v6 = v42;
      goto LABEL_14;
    }
  }
  return result;
}
