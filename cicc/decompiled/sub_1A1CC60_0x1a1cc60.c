// Function: sub_1A1CC60
// Address: 0x1a1cc60
//
_QWORD *__fastcall sub_1A1CC60(__int64 a1, __int64 a2, unsigned int a3, int a4, const __m128i *a5, int a6)
{
  int v6; // eax
  unsigned int v8; // ebx
  unsigned __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // r8
  int v13; // r9d
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // r11
  __int64 v20; // r10
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r15
  _QWORD *v24; // rax
  _QWORD *v25; // rbx
  __int64 v26; // [rsp+8h] [rbp-F8h]
  __int64 v28; // [rsp+18h] [rbp-E8h]
  __int64 v29; // [rsp+18h] [rbp-E8h]
  _QWORD *v30; // [rsp+18h] [rbp-E8h]
  __m128i v31; // [rsp+20h] [rbp-E0h] BYREF
  char v32; // [rsp+30h] [rbp-D0h]
  char v33; // [rsp+31h] [rbp-CFh]
  __m128i v34; // [rsp+40h] [rbp-C0h] BYREF
  char v35; // [rsp+50h] [rbp-B0h]
  char v36; // [rsp+51h] [rbp-AFh]
  __m128i v37; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v38; // [rsp+70h] [rbp-90h]
  __int64 *v39; // [rsp+80h] [rbp-80h] BYREF
  __int64 v40; // [rsp+88h] [rbp-78h]
  _WORD v41[56]; // [rsp+90h] [rbp-70h] BYREF

  v6 = a4 - a3;
  v8 = a3;
  v9 = a4 - a3;
  v10 = a2;
  if ( v9 != *(_QWORD *)(*(_QWORD *)a2 + 32LL) )
  {
    if ( v6 == 1 )
    {
      v34.m128i_i64[0] = (__int64)".extract";
      v36 = 1;
      v35 = 3;
      sub_14EC200(&v37, a5, &v34);
      v21 = sub_1643350(*(_QWORD **)(a1 + 24));
      v22 = sub_159C470(v21, v8, 0);
      v23 = v22;
      if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(v22 + 16) > 0x10u )
      {
        v41[0] = 257;
        v24 = sub_1648A60(56, 2u);
        v25 = v24;
        if ( v24 )
          sub_15FA320((__int64)v24, (_QWORD *)a2, v23, (__int64)&v39, 0);
        return sub_1A1C7B0((__int64 *)a1, v25, &v37);
      }
      else
      {
        return (_QWORD *)sub_15A37D0((_BYTE *)a2, v22, 0);
      }
    }
    else
    {
      v39 = (__int64 *)v41;
      v40 = 0x800000000LL;
      if ( v9 > 8 )
        sub_16CD150((__int64)&v39, v41, v9, 8, a2, a6);
      for ( ; a4 != v8; LODWORD(v40) = v40 + 1 )
      {
        v11 = sub_1643350(*(_QWORD **)(a1 + 24));
        v12 = sub_159C470(v11, v8, 0);
        v14 = (unsigned int)v40;
        if ( (unsigned int)v40 >= HIDWORD(v40) )
        {
          v26 = v12;
          sub_16CD150((__int64)&v39, v41, 0, 8, v12, v13);
          v14 = (unsigned int)v40;
          v12 = v26;
        }
        ++v8;
        v39[v14] = v12;
      }
      v33 = 1;
      v31.m128i_i64[0] = (__int64)".extract";
      v32 = 3;
      sub_14EC200(&v34, a5, &v31);
      v15 = sub_15A01B0(v39, (unsigned int)v40);
      v16 = sub_1599EF0(*(__int64 ***)a2);
      if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(v16 + 16) > 0x10u || *(_BYTE *)(v15 + 16) > 0x10u )
      {
        v29 = v16;
        v38 = 257;
        v18 = sub_1648A60(56, 3u);
        v19 = v18;
        if ( v18 )
        {
          v20 = v29;
          v30 = v18;
          sub_15FA660((__int64)v18, (_QWORD *)a2, v20, (_QWORD *)v15, (__int64)&v37, 0);
          v19 = v30;
        }
        v10 = (__int64)sub_1A1C7B0((__int64 *)a1, v19, &v34);
      }
      else
      {
        v10 = sub_15A3950(a2, v16, (_BYTE *)v15, 0);
      }
      if ( v39 != (__int64 *)v41 )
      {
        v28 = v10;
        _libc_free((unsigned __int64)v39);
        return (_QWORD *)v28;
      }
    }
  }
  return (_QWORD *)v10;
}
