// Function: sub_12545D0
// Address: 0x12545d0
//
__int64 __fastcall sub_12545D0(__int64 a1, int a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v4; // r15
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // rcx

  v4 = a1 + 8;
  *(_QWORD *)(a1 + 16) = 0;
  *(_BYTE *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 40) = a2;
  *(_QWORD *)a1 = &unk_49E6890;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  result = sub_2241130(a1 + 8, 0, 0, "Stream Error: ", 14);
  switch ( a2 )
  {
    case 0:
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 16)) <= 0x21 )
        goto LABEL_16;
      result = sub_2241490(v4, "An unspecified error has occurred.", 34, v8);
      break;
    case 1:
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 16)) <= 0x3A )
        goto LABEL_16;
      result = sub_2241490(v4, "The stream is too short to perform the requested operation.", 59, v8);
      break;
    case 2:
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 16)) <= 0x3B )
        goto LABEL_16;
      result = sub_2241490(v4, "The buffer size is not a multiple of the array element size.", 60, v8);
      break;
    case 3:
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 16)) <= 0x36 )
        goto LABEL_16;
      result = sub_2241490(v4, "The specified offset is invalid for the current stream.", 55, v8);
      break;
    case 4:
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 16)) <= 0x28 )
        goto LABEL_16;
      result = sub_2241490(v4, "An I/O error occurred on the file system.", 41, v8);
      break;
    default:
      break;
  }
  if ( a4 )
  {
    if ( *(_QWORD *)(a1 + 16) == 0x3FFFFFFFFFFFFFFFLL
      || *(_QWORD *)(a1 + 16) == 4611686018427387902LL
      || (sub_2241490(v4, "  ", 2, v8), a4 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 16)) )
    {
LABEL_16:
      sub_4262D8((__int64)"basic_string::append");
    }
    return sub_2241490(v4, a3, a4, v9);
  }
  return result;
}
