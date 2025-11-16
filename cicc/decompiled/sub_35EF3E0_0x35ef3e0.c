// Function: sub_35EF3E0
// Address: 0x35ef3e0
//
void __fastcall sub_35EF3E0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  __int64 v6; // r12
  size_t v9; // rax
  __int64 v10; // r12
  const char *v11; // rdx
  _BYTE *v12; // rax
  size_t v13; // rdx
  char *v14; // rsi
  void *v15; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // rdx
  _QWORD *v18; // rdx
  _QWORD *v19; // rdx
  _DWORD *v20; // rdx
  _QWORD *v21; // rdx
  _DWORD *v22; // rdx
  _DWORD *v23; // rdx
  _BYTE *v24; // rax
  _BYTE *v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rdx
  _BYTE *v29; // rax
  __int64 v30; // rdx
  __m128i v31[2]; // [rsp+0h] [rbp-D0h] BYREF
  __m128i *v32; // [rsp+20h] [rbp-B0h] BYREF
  __int16 v33; // [rsp+40h] [rbp-90h]
  __m128i v34; // [rsp+50h] [rbp-80h] BYREF
  __int64 v35; // [rsp+60h] [rbp-70h] BYREF

  v5 = 16LL * a3;
  if ( !a5 )
    goto LABEL_84;
  v6 = v5;
  v9 = strlen((const char *)a5);
  v10 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + v6 + 8);
  if ( v9 == 3 )
  {
    if ( *(_WORD *)a5 == 25971 && *(_BYTE *)(a5 + 2) == 109 )
    {
      switch ( (int)v10 )
      {
        case 0:
          return;
        case 2:
          v19 = *(_QWORD **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v19 > 7u )
          {
            *v19 = 0x646578616C65722ELL;
            *(_QWORD *)(a4 + 32) += 8LL;
            return;
          }
          v13 = 8;
          v14 = ".relaxed";
          goto LABEL_51;
        case 4:
          v17 = *(_QWORD **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v17 > 7u )
          {
            *v17 = 0x657269757163612ELL;
            *(_QWORD *)(a4 + 32) += 8LL;
            return;
          }
          v13 = 8;
          v14 = ".acquire";
          goto LABEL_51;
        case 5:
          v18 = *(_QWORD **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v18 > 7u )
          {
            *v18 = 0x657361656C65722ELL;
            *(_QWORD *)(a4 + 32) += 8LL;
            return;
          }
          v13 = 8;
          v14 = ".release";
          goto LABEL_51;
        case 8:
          v16 = *(_QWORD *)(a4 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v16) > 8 )
          {
            *(_BYTE *)(v16 + 8) = 101;
            *(_QWORD *)v16 = 0x6C6974616C6F762ELL;
            *(_QWORD *)(a4 + 32) += 9LL;
            return;
          }
          v13 = 9;
          v14 = ".volatile";
          goto LABEL_51;
        case 9:
          v15 = *(void **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v15 > 0xCu )
          {
            qmemcpy(v15, ".mmio.relaxed", 13);
            *(_QWORD *)(a4 + 32) += 13LL;
            return;
          }
          v13 = 13;
          v14 = ".mmio.relaxed";
          break;
        default:
          sub_35EDF20((__int64)v31, v10);
          v11 = "NVPTX LdStCode Printer does not support \"{}\" sem modifier. Loads/Stores cannot be AcquireRelease or Se"
                "quentiallyConsistent.";
          goto LABEL_19;
      }
      goto LABEL_51;
    }
    if ( *(_WORD *)a5 == 25974 && *(_BYTE *)(a5 + 2) == 99 )
    {
      switch ( (_DWORD)v10 )
      {
        case 4:
          v30 = *(_QWORD *)(a4 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v30) > 2 )
          {
            *(_BYTE *)(v30 + 2) = 52;
            *(_WORD *)v30 = 30254;
            *(_QWORD *)(a4 + 32) += 3LL;
            return;
          }
          v13 = 3;
          v14 = ".v4";
          break;
        case 8:
          v28 = *(_QWORD *)(a4 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v28) > 2 )
          {
            *(_BYTE *)(v28 + 2) = 56;
            *(_WORD *)v28 = 30254;
            *(_QWORD *)(a4 + 32) += 3LL;
            return;
          }
          v13 = 3;
          v14 = ".v8";
          break;
        case 2:
          v27 = *(_QWORD *)(a4 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v27) > 2 )
          {
            *(_BYTE *)(v27 + 2) = 50;
            *(_WORD *)v27 = 30254;
            *(_QWORD *)(a4 + 32) += 3LL;
            return;
          }
          v13 = 3;
          v14 = ".v2";
          break;
        default:
          return;
      }
LABEL_51:
      sub_CB6200(a4, (unsigned __int8 *)v14, v13);
      return;
    }
    goto LABEL_84;
  }
  if ( v9 != 5 )
  {
    if ( v9 == 4 && *(_DWORD *)a5 == 1852270963 )
    {
      if ( (_DWORD)v10 == 2 )
      {
        v29 = *(_BYTE **)(a4 + 32);
        v13 = 1;
        v14 = "f";
        if ( *(_BYTE **)(a4 + 24) != v29 )
        {
          *v29 = 102;
          ++*(_QWORD *)(a4 + 32);
          return;
        }
        goto LABEL_51;
      }
      if ( (int)v10 > 2 )
      {
        if ( (_DWORD)v10 == 3 )
        {
          v26 = *(_BYTE **)(a4 + 32);
          v13 = 1;
          v14 = "b";
          if ( *(_BYTE **)(a4 + 24) != v26 )
          {
            *v26 = 98;
            ++*(_QWORD *)(a4 + 32);
            return;
          }
          goto LABEL_51;
        }
      }
      else
      {
        if ( !(_DWORD)v10 )
        {
          v12 = *(_BYTE **)(a4 + 32);
          v13 = 1;
          v14 = (char *)"u";
          if ( *(_BYTE **)(a4 + 24) != v12 )
          {
            *v12 = 117;
            ++*(_QWORD *)(a4 + 32);
            return;
          }
          goto LABEL_51;
        }
        if ( (_DWORD)v10 == 1 )
        {
          v25 = *(_BYTE **)(a4 + 32);
          v13 = 1;
          v14 = "s";
          if ( *(_BYTE **)(a4 + 24) != v25 )
          {
            *v25 = 115;
            ++*(_QWORD *)(a4 + 32);
            return;
          }
          goto LABEL_51;
        }
      }
    }
LABEL_84:
    BUG();
  }
  if ( *(_DWORD *)a5 == 1886348147 && *(_BYTE *)(a5 + 4) == 101 )
  {
    switch ( (int)v10 )
    {
      case 0:
        return;
      case 1:
        v23 = *(_DWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v23 > 3u )
        {
          *v23 = 1635017518;
          *(_QWORD *)(a4 + 32) += 4LL;
          return;
        }
        v13 = 4;
        v14 = ".cta";
        goto LABEL_51;
      case 2:
        v21 = *(_QWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v21 > 7u )
        {
          *v21 = 0x72657473756C632ELL;
          *(_QWORD *)(a4 + 32) += 8LL;
          return;
        }
        v13 = 8;
        v14 = ".cluster";
        goto LABEL_51;
      case 3:
        v20 = *(_DWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v20 > 3u )
        {
          *v20 = 1970300718;
          *(_QWORD *)(a4 + 32) += 4LL;
          return;
        }
        v13 = 4;
        v14 = ".gpu";
        goto LABEL_51;
      case 4:
        v22 = *(_DWORD **)(a4 + 32);
        if ( *(_QWORD *)(a4 + 24) - (_QWORD)v22 > 3u )
        {
          *v22 = 1937339182;
          *(_QWORD *)(a4 + 32) += 4LL;
          return;
        }
        v13 = 4;
        v14 = ".sys";
        break;
      default:
        sub_35EE190((__int64)v31, v10);
        v11 = "NVPTX LdStCode Printer does not support \"{}\" sco modifier.";
        goto LABEL_19;
    }
    goto LABEL_51;
  }
  if ( *(_DWORD *)a5 != 1935959137 || *(_BYTE *)(a5 + 4) != 112 )
    goto LABEL_84;
  if ( (_DWORD)v10 != 1 )
  {
    if ( (unsigned int)v10 <= 1 )
      return;
    if ( (unsigned int)v10 > 5 )
    {
      if ( (_DWORD)v10 != 101 )
        goto LABEL_18;
    }
    else if ( (_DWORD)v10 == 2 )
    {
LABEL_18:
      sub_35EE2E0((__int64)v31, v10);
      v11 = "NVPTX LdStCode Printer does not support \"{}\" addsp modifier.";
LABEL_19:
      sub_35EF270(&v34, 1, v11, v31);
      v33 = 263;
      v32 = &v34;
      sub_C64D30((__int64)&v32, 1u);
    }
  }
  v24 = *(_BYTE **)(a4 + 32);
  if ( *(_BYTE **)(a4 + 24) == v24 )
  {
    a4 = sub_CB6200(a4, (unsigned __int8 *)".", 1u);
  }
  else
  {
    *v24 = 46;
    ++*(_QWORD *)(a4 + 32);
  }
  sub_35EE2E0((__int64)&v34, v10);
  sub_CB6200(a4, (unsigned __int8 *)v34.m128i_i64[0], v34.m128i_u64[1]);
  if ( (__int64 *)v34.m128i_i64[0] != &v35 )
    j_j___libc_free_0(v34.m128i_u64[0]);
}
