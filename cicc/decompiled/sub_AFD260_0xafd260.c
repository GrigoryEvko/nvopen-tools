// Function: sub_AFD260
// Address: 0xafd260
//
__int64 __fastcall sub_AFD260(__int64 a1, __int64 *a2, __int64 **a3)
{
  int v4; // r15d
  __int64 v6; // r14
  __int64 v7; // rax
  __int8 *v8; // rax
  __int8 *v9; // r14
  int v10; // ecx
  int v11; // r8d
  __int64 v12; // r14
  unsigned int v13; // ecx
  __int64 *v14; // r15
  __int64 v15; // r9
  _BYTE *v16; // r13
  __int16 v17; // ax
  __int64 v18; // rax
  __int16 v19; // ax
  __int64 v20; // rax
  _BYTE *v21; // rax
  __int64 *v22; // rax
  __int64 v23; // [rsp+8h] [rbp-178h]
  int v24; // [rsp+14h] [rbp-16Ch]
  __int64 *v25; // [rsp+18h] [rbp-168h]
  __int64 v26; // [rsp+18h] [rbp-168h]
  unsigned int v27; // [rsp+20h] [rbp-160h]
  int v28; // [rsp+28h] [rbp-158h]
  void *src; // [rsp+30h] [rbp-150h]
  __int64 v30; // [rsp+38h] [rbp-148h]
  __int64 v31; // [rsp+48h] [rbp-138h] BYREF
  __int64 v32; // [rsp+50h] [rbp-130h] BYREF
  __int64 v33; // [rsp+58h] [rbp-128h]
  int v34; // [rsp+60h] [rbp-120h] BYREF
  __int64 v35; // [rsp+68h] [rbp-118h] BYREF
  __int64 v36; // [rsp+70h] [rbp-110h] BYREF
  int v37; // [rsp+78h] [rbp-108h] BYREF
  _BYTE *v38; // [rsp+80h] [rbp-100h] BYREF
  __int64 v39[3]; // [rsp+88h] [rbp-F8h] BYREF
  int v40; // [rsp+A0h] [rbp-E0h]
  __int64 v41; // [rsp+A4h] [rbp-DCh]
  __int64 v42; // [rsp+ACh] [rbp-D4h]
  int v43; // [rsp+B4h] [rbp-CCh] BYREF
  __int64 v44; // [rsp+B8h] [rbp-C8h]
  __int64 v45; // [rsp+C0h] [rbp-C0h]
  __m128i dest[4]; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned __int64 v47[7]; // [rsp+110h] [rbp-70h] BYREF
  __int64 (__fastcall *v48)(); // [rsp+148h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v30 = *(_QWORD *)(a1 + 8);
    v34 = (unsigned __int16)sub_AF18C0(*a2);
    v35 = sub_AF5140(v6, 2u);
    v7 = v6;
    if ( *(_BYTE *)v6 != 16 )
      v7 = *(_QWORD *)sub_A17150((_BYTE *)(v6 - 16));
    v36 = v7;
    v37 = *(_DWORD *)(v6 + 16);
    v38 = (_BYTE *)*((_QWORD *)sub_A17150((_BYTE *)(v6 - 16)) + 1);
    v39[0] = *((_QWORD *)sub_A17150((_BYTE *)(v6 - 16)) + 3);
    v39[1] = *(_QWORD *)(v6 + 24);
    v39[2] = *(_QWORD *)(v6 + 32);
    v40 = sub_AF18D0(v6);
    v33 = *(_QWORD *)(v6 + 44);
    dest[0].m128i_i64[0] = v33;
    v41 = v33;
    v42 = sub_AF2E40(v6);
    v43 = *(_DWORD *)(v6 + 20);
    v44 = *((_QWORD *)sub_A17150((_BYTE *)(v6 - 16)) + 4);
    v45 = *((_QWORD *)sub_A17150((_BYTE *)(v6 - 16)) + 5);
    if ( v34 == 13 && v35 && v38 && *v38 == 14 && sub_AF5140((__int64)v38, 7u) )
    {
      memset(dest, 0, sizeof(dest));
      memset(v47, 0, sizeof(v47));
      v48 = sub_C64CA0;
      v31 = 0;
      v8 = sub_AF8740(dest, &v31, dest[0].m128i_i8, (unsigned __int64)v47, v35);
      v32 = v31;
      v9 = sub_AF70F0(dest, &v32, v8, (unsigned __int64)v47, (__int64)v38);
      if ( v32 )
      {
        v26 = v32;
        sub_AF1140(dest[0].m128i_i8, v9, (char *)v47);
        sub_AC2A10(v47, dest);
        v10 = sub_AF1490(v47, v9 - (__int8 *)dest + v26);
      }
      else
      {
        v10 = sub_AC25F0(dest, v9 - (__int8 *)dest, (__int64)v48);
      }
    }
    else
    {
      v10 = sub_AF95C0(&v34, &v35, &v36, &v37, (__int64 *)&v38, v39, &v43);
    }
    v11 = v4 - 1;
    v12 = *a2;
    v13 = (v4 - 1) & v10;
    v14 = (__int64 *)(v30 + 8LL * v13);
    v15 = *v14;
    if ( *a2 != *v14 )
    {
      v24 = 1;
      v25 = 0;
      while ( 1 )
      {
        if ( v15 != -8192 )
        {
          v23 = v15;
          if ( v15 != -4096 )
          {
            v27 = v13;
            v28 = v11;
            src = (void *)sub_AF5140(v12, 2u);
            v16 = (_BYTE *)*((_QWORD *)sub_A17150((_BYTE *)(v12 - 16)) + 1);
            v17 = sub_AF18C0(v12);
            v11 = v28;
            v13 = v27;
            if ( src != 0 && v17 == 13 )
            {
              if ( v16 )
              {
                if ( *v16 == 14 )
                {
                  v18 = sub_AF5140((__int64)v16, 7u);
                  v11 = v28;
                  v13 = v27;
                  if ( v18 )
                  {
                    v19 = sub_AF18C0(v23);
                    v11 = v28;
                    v13 = v27;
                    if ( v19 == 13 )
                    {
                      v20 = sub_AF5140(v23, 2u);
                      v11 = v28;
                      v13 = v27;
                      if ( src == (void *)v20 )
                      {
                        v21 = sub_A17150((_BYTE *)(v23 - 16));
                        v11 = v28;
                        v13 = v27;
                        if ( v16 == *((_BYTE **)v21 + 1) )
                          break;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if ( *v14 == -4096 )
        {
          if ( v25 )
            v14 = v25;
          *a3 = v14;
          return 0;
        }
        if ( *v14 != -8192 || v25 )
          v14 = v25;
        v12 = *a2;
        v13 = v11 & (v24 + v13);
        v22 = (__int64 *)(v30 + 8LL * v13);
        v15 = *v22;
        if ( *v22 == *a2 )
        {
          v14 = (__int64 *)(v30 + 8LL * v13);
          break;
        }
        v25 = v14;
        v14 = (__int64 *)(v30 + 8LL * v13);
        ++v24;
      }
    }
    *a3 = v14;
    return 1;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
