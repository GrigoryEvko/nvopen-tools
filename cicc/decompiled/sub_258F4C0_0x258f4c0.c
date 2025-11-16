// Function: sub_258F4C0
// Address: 0x258f4c0
//
__m128i *__fastcall sub_258F4C0(__m128i *a1, __int64 a2, _QWORD *a3)
{
  unsigned int v3; // r14d
  char v5; // bl
  char v6; // al
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rcx
  int v10; // r8d
  unsigned int v11; // r8d
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rcx
  int v15; // r9d
  char *v16; // rdx
  const char *v17; // r9
  const char *v19; // [rsp+28h] [rbp-198h]
  unsigned int v20; // [rsp+30h] [rbp-190h]
  const char *v21; // [rsp+30h] [rbp-190h]
  const char *v22; // [rsp+38h] [rbp-188h]
  char v23; // [rsp+4Fh] [rbp-171h] BYREF
  unsigned __int64 v24[4]; // [rsp+50h] [rbp-170h] BYREF
  __m128i v25[2]; // [rsp+70h] [rbp-150h] BYREF
  __m128i v26[2]; // [rsp+90h] [rbp-130h] BYREF
  __m128i v27[2]; // [rsp+B0h] [rbp-110h] BYREF
  char *v28; // [rsp+D0h] [rbp-F0h] BYREF
  int v29; // [rsp+D8h] [rbp-E8h]
  char v30; // [rsp+E0h] [rbp-E0h] BYREF
  __m128i v31[2]; // [rsp+F0h] [rbp-D0h] BYREF
  __m128i v32[2]; // [rsp+110h] [rbp-B0h] BYREF
  char *v33; // [rsp+130h] [rbp-90h] BYREF
  int v34; // [rsp+138h] [rbp-88h]
  char v35; // [rsp+140h] [rbp-80h] BYREF
  __m128i v36[2]; // [rsp+150h] [rbp-70h] BYREF
  __m128i v37[5]; // [rsp+170h] [rbp-50h] BYREF

  v3 = *(_DWORD *)(a2 + 108);
  if ( v3 )
  {
    v5 = 0;
    v22 = " [non-null is unknown]";
    if ( a3 )
    {
      v6 = sub_258F340(a3, a2, (__m128i *)(a2 + 72), 2, &v23, 0, 0);
      v3 = *(_DWORD *)(a2 + 108);
      v5 = v6;
      v22 = byte_3F871B3;
    }
    if ( v3 <= 9 )
    {
      v8 = 1;
    }
    else if ( v3 <= 0x63 )
    {
      v8 = 2;
    }
    else if ( v3 <= 0x3E7 )
    {
      v8 = 3;
    }
    else
    {
      v7 = v3;
      if ( v3 <= 0x270F )
      {
        v8 = 4;
      }
      else
      {
        LODWORD(v8) = 1;
        while ( 1 )
        {
          v9 = v7;
          v10 = v8;
          v8 = (unsigned int)(v8 + 4);
          v7 /= 0x2710u;
          if ( v9 <= 0x1869F )
            break;
          if ( (unsigned int)v7 <= 0x63 )
          {
            v8 = (unsigned int)(v10 + 5);
            break;
          }
          if ( (unsigned int)v7 <= 0x3E7 )
          {
            v8 = (unsigned int)(v10 + 6);
            break;
          }
          if ( (unsigned int)v7 <= 0x270F )
          {
            v8 = (unsigned int)(v10 + 7);
            break;
          }
        }
      }
    }
    v33 = &v35;
    sub_2240A50((__int64 *)&v33, v8, 0);
    sub_2554A60(v33, v34, v3);
    v11 = *(_DWORD *)(a2 + 104);
    if ( v11 <= 9 )
    {
      v13 = 1;
    }
    else if ( v11 <= 0x63 )
    {
      v13 = 2;
    }
    else if ( v11 <= 0x3E7 )
    {
      v13 = 3;
    }
    else
    {
      v12 = v11;
      if ( v11 <= 0x270F )
      {
        v13 = 4;
      }
      else
      {
        LODWORD(v13) = 1;
        while ( 1 )
        {
          v14 = v12;
          v15 = v13;
          v13 = (unsigned int)(v13 + 4);
          v12 /= 0x2710u;
          if ( v14 <= 0x1869F )
            break;
          if ( (unsigned int)v12 <= 0x63 )
          {
            v13 = (unsigned int)(v15 + 5);
            break;
          }
          if ( (unsigned int)v12 <= 0x3E7 )
          {
            v13 = (unsigned int)(v15 + 6);
            break;
          }
          if ( (unsigned int)v12 <= 0x270F )
          {
            v13 = (unsigned int)(v15 + 7);
            break;
          }
        }
      }
    }
    v20 = *(_DWORD *)(a2 + 104);
    v28 = &v30;
    sub_2240A50((__int64 *)&v28, v13, 0);
    sub_2554A60(v28, v29, v20);
    v16 = (char *)byte_3F871B3;
    v17 = "_globally";
    if ( !*(_BYTE *)(a2 + 169) )
      v17 = byte_3F871B3;
    if ( !v5 )
      v16 = "_or_null";
    v19 = v17;
    v21 = v16;
    sub_253C590((__int64 *)v24, "dereferenceable");
    sub_94F930(v25, (__int64)v24, v21);
    sub_94F930(v26, (__int64)v25, v19);
    sub_94F930(v27, (__int64)v26, "<");
    sub_8FD5D0(v31, (__int64)v27, &v28);
    sub_94F930(v32, (__int64)v31, "-");
    sub_8FD5D0(v36, (__int64)v32, &v33);
    sub_94F930(v37, (__int64)v36, ">");
    sub_94F930(a1, (__int64)v37, v22);
    sub_2240A30((unsigned __int64 *)v37);
    sub_2240A30((unsigned __int64 *)v36);
    sub_2240A30((unsigned __int64 *)v32);
    sub_2240A30((unsigned __int64 *)v31);
    sub_2240A30((unsigned __int64 *)v27);
    sub_2240A30((unsigned __int64 *)v26);
    sub_2240A30((unsigned __int64 *)v25);
    sub_2240A30(v24);
    sub_2240A30((unsigned __int64 *)&v28);
    sub_2240A30((unsigned __int64 *)&v33);
  }
  else
  {
    sub_253C590(a1->m128i_i64, "unknown-dereferenceable");
  }
  return a1;
}
