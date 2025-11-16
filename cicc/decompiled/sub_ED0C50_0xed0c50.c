// Function: sub_ED0C50
// Address: 0xed0c50
//
__int64 *__fastcall sub_ED0C50(__int64 *a1, int a2, __int64 a3)
{
  _QWORD *v4; // rax
  size_t v6; // rdx
  unsigned __int8 *v7; // rsi
  __int64 v8; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-90h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-80h] BYREF
  _QWORD v11[3]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v12; // [rsp+38h] [rbp-58h]
  _WORD *v13; // [rsp+40h] [rbp-50h]
  __int64 v14; // [rsp+48h] [rbp-48h]
  _QWORD *v15; // [rsp+50h] [rbp-40h]

  v14 = 0x100000000LL;
  v15 = v9;
  v9[0] = v10;
  v11[0] = &unk_49DD210;
  v9[1] = 0;
  LOBYTE(v10[0]) = 0;
  v11[1] = 0;
  v11[2] = 0;
  v12 = 0;
  v13 = 0;
  sub_CB5980((__int64)v11, 0, 0, 0);
  switch ( a2 )
  {
    case 0:
      sub_904010((__int64)v11, "success");
      break;
    case 1:
      sub_904010((__int64)v11, "end of File");
      break;
    case 2:
      sub_904010((__int64)v11, "unrecognized instrumentation profile encoding format");
      break;
    case 3:
      sub_904010((__int64)v11, "invalid instrumentation profile data (bad magic)");
      break;
    case 4:
      sub_904010((__int64)v11, "invalid instrumentation profile data (file header is corrupt)");
      break;
    case 5:
      sub_904010((__int64)v11, "unsupported instrumentation profile format version");
      break;
    case 6:
      sub_904010((__int64)v11, "unsupported instrumentation profile hash type");
      break;
    case 7:
      sub_904010((__int64)v11, "too much profile data");
      break;
    case 8:
      sub_904010((__int64)v11, "truncated profile data");
      break;
    case 9:
      sub_904010((__int64)v11, "malformed instrumentation profile data");
      break;
    case 10:
      sub_904010((__int64)v11, "debug info/binary for correlation is required");
      break;
    case 11:
      sub_904010((__int64)v11, "debug info/binary for correlation is not necessary");
      break;
    case 12:
      sub_904010((__int64)v11, "unable to correlate profile");
      break;
    case 13:
      sub_904010((__int64)v11, "no profile data available for function");
      break;
    case 14:
      sub_904010(
        (__int64)v11,
        "invalid profile created. Please file a bug at:  and include the profraw files that caused this error.");
      break;
    case 15:
      sub_904010((__int64)v11, "function control flow change detected (hash mismatch)");
      break;
    case 16:
      sub_904010((__int64)v11, "function basic block count change detected (counter mismatch)");
      break;
    case 17:
      sub_904010((__int64)v11, "function bitmap size change detected (bitmap size mismatch)");
      break;
    case 18:
      sub_904010((__int64)v11, "counter overflow");
      break;
    case 19:
      sub_904010((__int64)v11, "function value site count change detected (counter mismatch)");
      break;
    case 20:
      sub_904010((__int64)v11, "failed to compress data (zlib)");
      break;
    case 21:
      sub_904010((__int64)v11, "failed to uncompress data (zlib)");
      break;
    case 22:
      sub_904010((__int64)v11, "empty raw profile file");
      break;
    case 23:
      sub_904010((__int64)v11, "profile uses zlib compression but the profile reader was built without zlib support");
      break;
    case 24:
      sub_904010((__int64)v11, "raw profile version mismatch");
      break;
    case 25:
      sub_904010((__int64)v11, "excessively large counter value suggests corrupted profile data");
      break;
    default:
      break;
  }
  if ( *(_QWORD *)(a3 + 8) )
  {
    if ( (unsigned __int64)(v12 - (_QWORD)v13) <= 1 )
    {
      v8 = sub_CB6200((__int64)v11, (unsigned __int8 *)": ", 2u);
      sub_CB6200(v8, *(unsigned __int8 **)a3, *(_QWORD *)(a3 + 8));
    }
    else
    {
      *v13 = 8250;
      v6 = *(_QWORD *)(a3 + 8);
      v7 = *(unsigned __int8 **)a3;
      ++v13;
      sub_CB6200((__int64)v11, v7, v6);
    }
  }
  v4 = v15;
  *a1 = (__int64)(a1 + 2);
  sub_ED0570(a1, (_BYTE *)*v4, *v4 + v4[1]);
  v11[0] = &unk_49DD210;
  sub_CB5840((__int64)v11);
  if ( (_QWORD *)v9[0] != v10 )
    j_j___libc_free_0(v9[0], v10[0] + 1LL);
  return a1;
}
