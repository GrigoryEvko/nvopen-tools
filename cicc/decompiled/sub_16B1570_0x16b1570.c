// Function: sub_16B1570
// Address: 0x16b1570
//
__int64 __fastcall sub_16B1570(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        void (__fastcall *a4)(_BYTE *, size_t, __int64, __int64, _QWORD),
        __int64 a5,
        unsigned __int8 a6,
        char a7)
{
  unsigned int v7; // r15d
  _BYTE *v11; // rdi
  __int64 v12; // rsi
  size_t v13; // rsi
  __int64 v14; // rax
  int v15; // r12d
  __int64 v16; // rbx
  _BYTE *v17; // r15
  size_t v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  char *v21; // r15
  size_t v22; // rdx
  __int64 v23; // rdi
  size_t v24; // r11
  void *v25; // r9
  int v26; // edx
  int v27; // r8d
  void *v28; // [rsp+8h] [rbp-268h]
  int v29; // [rsp+14h] [rbp-25Ch]
  int v30; // [rsp+18h] [rbp-258h]
  size_t v31; // [rsp+18h] [rbp-258h]
  size_t v32; // [rsp+28h] [rbp-248h]
  __int64 v35; // [rsp+50h] [rbp-220h] BYREF
  __int64 v36; // [rsp+58h] [rbp-218h]
  _QWORD v37[2]; // [rsp+60h] [rbp-210h] BYREF
  _QWORD v38[2]; // [rsp+70h] [rbp-200h] BYREF
  _QWORD *v39; // [rsp+80h] [rbp-1F0h] BYREF
  char v40; // [rsp+90h] [rbp-1E0h]
  _QWORD *v41; // [rsp+A0h] [rbp-1D0h] BYREF
  __int16 v42; // [rsp+B0h] [rbp-1C0h]
  _QWORD *v43; // [rsp+C0h] [rbp-1B0h] BYREF
  __int16 v44; // [rsp+D0h] [rbp-1A0h]
  _BYTE v45[16]; // [rsp+E0h] [rbp-190h] BYREF
  __int16 v46; // [rsp+F0h] [rbp-180h]
  _QWORD *v47; // [rsp+100h] [rbp-170h]
  size_t v48; // [rsp+108h] [rbp-168h]
  _QWORD v49[2]; // [rsp+110h] [rbp-160h] BYREF
  char *s; // [rsp+120h] [rbp-150h] BYREF
  __int64 v51; // [rsp+128h] [rbp-148h]
  _BYTE v52[128]; // [rsp+130h] [rbp-140h] BYREF
  void *src; // [rsp+1B0h] [rbp-C0h] BYREF
  size_t n; // [rsp+1B8h] [rbp-B8h]
  _WORD v55[88]; // [rsp+1C0h] [rbp-B0h] BYREF

  v7 = 0;
  v35 = a1;
  v36 = a2;
  v55[0] = 261;
  src = &v35;
  sub_16C2DE0(&v39, &src, -1, 1, 0);
  if ( (v40 & 1) != 0 )
    return v7;
  v11 = (_BYTE *)v39[1];
  v12 = v39[2];
  v48 = 0;
  v32 = v12 - (_QWORD)v11;
  v47 = v49;
  LOBYTE(v49[0]) = 0;
  v13 = v12 - (_QWORD)v11;
  if ( !(unsigned __int8)sub_16BA250() )
  {
    if ( v32 > 2 && *v11 == 0xEF && v11[1] == 0xBB && v11[2] == 0xBF )
    {
      v11 += 3;
      v13 = v32 - 3;
    }
    goto LABEL_7;
  }
  v7 = sub_16BA290(v11, v13);
  if ( (_BYTE)v7 )
  {
    v11 = v47;
    v13 = v48;
LABEL_7:
    a4(v11, v13, a3, a5, a6);
    if ( a7 && *(_DWORD *)(a5 + 8) )
    {
      v14 = 0;
      v15 = 0;
      do
      {
        v16 = 8 * v14;
        v17 = *(_BYTE **)(*(_QWORD *)a5 + 8 * v14);
        if ( v17 && *v17 == 64 )
        {
          v18 = strlen(*(const char **)(*(_QWORD *)a5 + 8 * v14));
          if ( v18 )
          {
            --v18;
            ++v17;
          }
          v37[1] = v18;
          v37[0] = v17;
          v55[0] = 261;
          src = v37;
          if ( (unsigned __int8)sub_16C4FD0(&src, 2) )
          {
            v52[0] = 64;
            s = v52;
            v51 = 0x8000000001LL;
            v55[0] = 261;
            src = &v35;
            if ( (unsigned __int8)sub_16C4FD0(&src, 2) )
            {
              src = v55;
              n = 0x8000000000LL;
              sub_16C56A0(&src);
              v23 = (unsigned int)v51;
              v24 = (unsigned int)n;
              v25 = src;
              v26 = v51;
              v27 = n;
              if ( (unsigned int)n > HIDWORD(v51) - (unsigned __int64)(unsigned int)v51 )
              {
                v28 = src;
                v29 = n;
                v31 = (unsigned int)n;
                sub_16CD150(&s, v52, (unsigned int)n + (unsigned __int64)(unsigned int)v51, 1);
                v23 = (unsigned int)v51;
                v25 = v28;
                v27 = v29;
                v24 = v31;
                v26 = v51;
              }
              if ( v27 )
              {
                v30 = v27;
                memcpy(&s[v23], v25, v24);
                v26 = v51;
                v27 = v30;
              }
              LODWORD(v51) = v26 + v27;
              if ( src != v55 )
                _libc_free((unsigned __int64)src);
            }
            v55[0] = 257;
            v46 = 257;
            v44 = 261;
            v43 = v37;
            v38[0] = sub_16C41E0(v35, v36, 2);
            v38[1] = v19;
            v42 = 261;
            v41 = v38;
            sub_16C4D40(&s, &v41, &v43, v45, &src);
            v20 = (unsigned int)v51;
            if ( (unsigned int)v51 >= HIDWORD(v51) )
            {
              sub_16CD150(&s, v52, 0, 1);
              v20 = (unsigned int)v51;
            }
            s[v20] = 0;
            v21 = s;
            v22 = 0;
            if ( s )
              v22 = strlen(s);
            *(_QWORD *)(*(_QWORD *)a5 + v16) = sub_16D3940(a3, v21, v22);
            if ( s != v52 )
              _libc_free((unsigned __int64)s);
          }
        }
        v14 = (unsigned int)(v15 + 1);
        v15 = v14;
      }
      while ( (unsigned int)v14 < *(_DWORD *)(a5 + 8) );
    }
    v7 = 1;
  }
  if ( v47 != v49 )
    j_j___libc_free_0(v47, v49[0] + 1LL);
  if ( (v40 & 1) == 0 && v39 )
    (*(void (__fastcall **)(_QWORD *))(*v39 + 8LL))(v39);
  return v7;
}
