// Function: sub_1AEE3A0
// Address: 0x1aee3a0
//
__int64 __fastcall sub_1AEE3A0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rax
  __int64 v10; // r14
  _QWORD *v11; // rax
  unsigned __int8 v12; // r15
  double v13; // xmm4_8
  double v14; // xmm5_8
  __int64 *v15; // rax
  unsigned int v16; // esi
  _QWORD *v17; // rdi
  _QWORD *v18; // rax
  _QWORD *v19; // rdi
  int v21; // esi
  int v22; // edx
  unsigned int v23; // esi
  int v24; // ecx
  unsigned int v25; // eax
  unsigned __int64 v26; // rax
  __int64 v27; // r14
  _QWORD *v28; // rax
  _QWORD *i; // rcx
  unsigned __int8 v31; // [rsp+Ch] [rbp-64h]
  int v32; // [rsp+Ch] [rbp-64h]
  _QWORD *v33; // [rsp+10h] [rbp-60h] BYREF
  __int64 *v34; // [rsp+18h] [rbp-58h] BYREF
  __int64 v35; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v36; // [rsp+28h] [rbp-48h]
  __int64 v37; // [rsp+30h] [rbp-40h]
  __int64 v38; // [rsp+38h] [rbp-38h]

  v9 = *(_QWORD *)(a1 + 48);
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  if ( *(_BYTE *)(v9 - 8) != 77 )
  {
    v31 = 0;
    v19 = 0;
    goto LABEL_15;
  }
  v31 = 0;
  v10 = *(_QWORD *)(v9 + 8);
  v11 = (_QWORD *)(v9 - 24);
  while ( 1 )
  {
    v33 = v11;
    v12 = sub_1AEE080((__int64)&v35, (__int64 *)&v33, &v34);
    v15 = v34;
    if ( v12 )
    {
      sub_164D160((__int64)v33, *v34, a2, a3, a4, a5, v13, v14, a8, a9);
      sub_15F20C0(v33);
      ++v35;
      if ( v37 )
      {
        v16 = 4 * v37;
        v17 = v36;
        if ( (unsigned int)(4 * v37) < 0x40 )
          v16 = 64;
        v18 = &v36[(unsigned int)v38];
        if ( v16 < (unsigned int)v38 )
        {
          if ( (_DWORD)v37 )
          {
            if ( (_DWORD)v37 == 1 )
            {
              v32 = 128;
              v27 = 1024;
            }
            else
            {
              _BitScanReverse(&v23, v37 - 1);
              v24 = 1 << (33 - (v23 ^ 0x1F));
              if ( v24 < 64 )
                v24 = 64;
              if ( (_DWORD)v38 == v24 )
              {
                v37 = 0;
                do
                {
                  if ( v17 )
                    *v17 = -8;
                  ++v17;
                }
                while ( v18 != v17 );
                goto LABEL_11;
              }
              v25 = 4 * v24;
              v26 = (((v25 / 3 + 1) | ((unsigned __int64)(v25 / 3 + 1) >> 1)) >> 2)
                  | (v25 / 3 + 1)
                  | ((unsigned __int64)(v25 / 3 + 1) >> 1)
                  | (((((v25 / 3 + 1) | ((unsigned __int64)(v25 / 3 + 1) >> 1)) >> 2)
                    | (v25 / 3 + 1)
                    | ((unsigned __int64)(v25 / 3 + 1) >> 1)) >> 4);
              v32 = ((v26 >> 8) | v26 | (((v26 >> 8) | v26) >> 16)) + 1;
              v27 = 8 * (((v26 >> 8) | v26 | (((v26 >> 8) | v26) >> 16)) + 1);
            }
            j___libc_free_0(v36);
            LODWORD(v38) = v32;
            v28 = (_QWORD *)sub_22077B0(v27);
            v37 = 0;
            v36 = v28;
            for ( i = &v28[(unsigned int)v38]; i != v28; ++v28 )
            {
              if ( v28 )
                *v28 = -8;
            }
          }
          else
          {
            j___libc_free_0(v36);
            v36 = 0;
            v37 = 0;
            LODWORD(v38) = 0;
          }
        }
        else
        {
          if ( v18 != v36 )
          {
            do
              *v17++ = -8;
            while ( v17 != v18 );
          }
          v37 = 0;
        }
      }
LABEL_11:
      v31 = v12;
      v10 = *(_QWORD *)(a1 + 48);
      goto LABEL_12;
    }
    v21 = v38;
    ++v35;
    v22 = v37 + 1;
    if ( 4 * ((int)v37 + 1) >= (unsigned int)(3 * v38) )
    {
      v21 = 2 * v38;
LABEL_22:
      sub_1AEE230((__int64)&v35, v21);
      sub_1AEE080((__int64)&v35, (__int64 *)&v33, &v34);
      v15 = v34;
      v22 = v37 + 1;
      goto LABEL_18;
    }
    if ( (int)v38 - HIDWORD(v37) - v22 <= (unsigned int)v38 >> 3 )
      goto LABEL_22;
LABEL_18:
    LODWORD(v37) = v22;
    if ( *v15 != -8 )
      --HIDWORD(v37);
    *v15 = (__int64)v33;
LABEL_12:
    v11 = (_QWORD *)(v10 - 24);
    if ( *(_BYTE *)(v10 - 8) != 77 )
      break;
    v10 = *(_QWORD *)(v10 + 8);
  }
  v19 = v36;
LABEL_15:
  j___libc_free_0(v19);
  return v31;
}
