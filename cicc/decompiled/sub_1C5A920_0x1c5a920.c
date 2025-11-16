// Function: sub_1C5A920
// Address: 0x1c5a920
//
__int64 __fastcall sub_1C5A920(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r12d
  _QWORD *v4; // rdi
  _QWORD *v5; // rdi
  __int64 v6; // rdx
  __int64 *v7; // r10
  int v8; // r11d
  unsigned int v9; // eax
  __int64 *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rax
  int v13; // esi
  int v14; // ecx
  __int64 *v15; // r15
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v23; // [rsp+8h] [rbp-C8h]
  __int64 v24; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v25; // [rsp+20h] [rbp-B0h] BYREF
  __int64 *v26; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v27; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v28; // [rsp+38h] [rbp-98h]
  __int64 v29; // [rsp+40h] [rbp-90h]
  __int64 v30; // [rsp+48h] [rbp-88h]
  __int64 v31[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD *v32; // [rsp+60h] [rbp-70h]
  __int64 v33; // [rsp+68h] [rbp-68h]
  __int64 v34; // [rsp+70h] [rbp-60h]
  __int64 v35; // [rsp+78h] [rbp-58h]
  _QWORD *v36; // [rsp+80h] [rbp-50h]
  _QWORD *v37; // [rsp+88h] [rbp-48h]
  __int64 v38; // [rsp+90h] [rbp-40h]
  _QWORD *v39; // [rsp+98h] [rbp-38h]

  v2 = 0;
  if ( *(_BYTE *)(a1 + 16) > 0x17u )
  {
    v24 = a1;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    v30 = 0;
    v31[0] = 0;
    v31[1] = 0;
    v32 = 0;
    v33 = 0;
    v34 = 0;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    v39 = 0;
    sub_1C55F90(v31, 0);
    v4 = v36;
    if ( v36 == (_QWORD *)(v38 - 8) )
    {
      sub_1C56080(v31, &v24);
      v5 = v36;
    }
    else
    {
      if ( v36 )
      {
        *v36 = v24;
        v4 = v36;
      }
      v5 = v4 + 1;
      v36 = v5;
    }
    v2 = 0;
    while ( v32 != v5 )
    {
      if ( v37 == v5 )
      {
        v25 = *(_QWORD *)(*(v39 - 1) + 504LL);
        j_j___libc_free_0(v5, 512);
        v21 = *--v39 + 512LL;
        v37 = (_QWORD *)*v39;
        v38 = v21;
        v36 = v37 + 63;
      }
      else
      {
        v12 = *(v5 - 1);
        v36 = v5 - 1;
        v25 = v12;
      }
      v13 = v30;
      if ( (_DWORD)v30 )
      {
        v6 = v25;
        v7 = 0;
        v8 = 1;
        v9 = (v30 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v10 = (__int64 *)(v28 + 8LL * v9);
        v11 = *v10;
        if ( v25 == *v10 )
        {
LABEL_9:
          v5 = v36;
          continue;
        }
        while ( v11 != -8 )
        {
          if ( v7 || v11 != -16 )
            v10 = v7;
          v9 = (v30 - 1) & (v8 + v9);
          v15 = (__int64 *)(v28 + 8LL * v9);
          v11 = *v15;
          if ( v25 == *v15 )
            goto LABEL_9;
          ++v8;
          v7 = v10;
          v10 = (__int64 *)(v28 + 8LL * v9);
        }
        if ( !v7 )
          v7 = v10;
        ++v27;
        v14 = v29 + 1;
        if ( 4 * ((int)v29 + 1) < (unsigned int)(3 * v30) )
        {
          if ( (int)v30 - HIDWORD(v29) - v14 > (unsigned int)v30 >> 3 )
            goto LABEL_26;
          goto LABEL_16;
        }
      }
      else
      {
        ++v27;
      }
      v13 = 2 * v30;
LABEL_16:
      sub_1467110((__int64)&v27, v13);
      sub_1463A20((__int64)&v27, &v25, &v26);
      v7 = v26;
      v6 = v25;
      v14 = v29 + 1;
LABEL_26:
      LODWORD(v29) = v14;
      if ( *v7 != -8 )
        --HIDWORD(v29);
      ++v2;
      *v7 = v6;
      if ( v2 > a2 )
        break;
      v16 = v25;
      v5 = v36;
      if ( (*(_DWORD *)(v25 + 20) & 0xFFFFFFF) != 0 )
      {
        v17 = 0;
        v18 = 24LL * ((*(_DWORD *)(v25 + 20) & 0xFFFFFFFu) - 1);
        while ( 1 )
        {
          if ( (*(_BYTE *)(v16 + 23) & 0x40) != 0 )
            v19 = *(_QWORD *)(v16 - 8);
          else
            v19 = v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF);
          v20 = *(_QWORD *)(v19 + v17);
          if ( *(_BYTE *)(v20 + 16) > 0x17u )
          {
            v26 = (__int64 *)v20;
            if ( v5 == (_QWORD *)(v38 - 8) )
            {
              v23 = v18;
              sub_1C56080(v31, &v26);
              v5 = v36;
              v18 = v23;
            }
            else
            {
              if ( v5 )
              {
                *v5 = v20;
                v5 = v36;
              }
              v36 = ++v5;
            }
          }
          if ( v18 == v17 )
            break;
          v16 = v25;
          v17 += 24;
        }
      }
    }
    sub_1C55A80(v31);
    j___libc_free_0(v28);
  }
  return v2;
}
