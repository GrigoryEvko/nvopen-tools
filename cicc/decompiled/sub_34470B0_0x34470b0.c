// Function: sub_34470B0
// Address: 0x34470b0
//
__int64 __fastcall sub_34470B0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v4; // r14
  __int16 v5; // ax
  __int64 result; // rax
  __int64 v7; // rax
  bool v8; // zf
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  _QWORD v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = (_QWORD *)(a2 + 72);
  *(_BYTE *)(a1 + 32) = sub_B49B80(a2, a3, 54) & 1 | *(_BYTE *)(a1 + 32) & 0xFE;
  *(_BYTE *)(a1 + 32) = (2 * (sub_B49B80(a2, a3, 79) & 1)) | *(_BYTE *)(a1 + 32) & 0xFD;
  *(_BYTE *)(a1 + 32) = (4 * (sub_B49B80(a2, a3, 28) & 1)) | *(_BYTE *)(a1 + 32) & 0xFB;
  *(_BYTE *)(a1 + 32) = (8 * (sub_B49B80(a2, a3, 15) & 1)) | *(_BYTE *)(a1 + 32) & 0xF7;
  *(_BYTE *)(a1 + 32) = (16 * (sub_B49B80(a2, a3, 85) & 1)) | *(_BYTE *)(a1 + 32) & 0xEF;
  *(_BYTE *)(a1 + 32) = (32 * (sub_B49B80(a2, a3, 21) & 1)) | *(_BYTE *)(a1 + 32) & 0xDF;
  *(_BYTE *)(a1 + 32) = ((sub_B49B80(a2, a3, 81) & 1) << 6) | *(_BYTE *)(a1 + 32) & 0xBF;
  *(_BYTE *)(a1 + 33) = (2 * (sub_B49B80(a2, a3, 84) & 1)) | *(_BYTE *)(a1 + 33) & 0xFD;
  *(_BYTE *)(a1 + 33) = sub_B49B80(a2, a3, 83) & 1 | *(_BYTE *)(a1 + 33) & 0xFE;
  *(_BYTE *)(a1 + 33) = (4 * (sub_B49B80(a2, a3, 52) & 1)) | *(_BYTE *)(a1 + 33) & 0xFB;
  *(_BYTE *)(a1 + 33) = (8 * (sub_B49B80(a2, a3, 75) & 1)) | *(_BYTE *)(a1 + 33) & 0xF7;
  *(_BYTE *)(a1 + 33) = (16 * (sub_B49B80(a2, a3, 73) & 1)) | *(_BYTE *)(a1 + 33) & 0xEF;
  *(_BYTE *)(a1 + 33) = (32 * (sub_B49B80(a2, a3, 74) & 1)) | *(_BYTE *)(a1 + 33) & 0xDF;
  v5 = sub_A74860((_QWORD *)(a2 + 72), a3);
  *(_QWORD *)(a1 + 40) = 0;
  *(_WORD *)(a1 + 34) = v5;
  if ( (*(_BYTE *)(a1 + 32) & 0x40) != 0 )
  {
    v7 = sub_A748A0(v4, a3);
    if ( !v7 )
    {
      v13 = *(_QWORD *)(a2 - 32);
      if ( v13 )
      {
        if ( !*(_BYTE *)v13 && *(_QWORD *)(v13 + 24) == *(_QWORD *)(a2 + 80) )
        {
          v14[0] = *(_QWORD *)(v13 + 120);
          v7 = sub_A748A0(v14, a3);
        }
      }
    }
    v8 = *(_BYTE *)(a1 + 35) == 0;
    *(_QWORD *)(a1 + 40) = v7;
    if ( v8 )
    {
      *(_WORD *)(a1 + 34) = sub_A74840(v4, a3);
      result = *(unsigned __int8 *)(a1 + 33);
      if ( (result & 2) != 0 )
        goto LABEL_9;
LABEL_3:
      if ( (result & 1) != 0 )
        goto LABEL_11;
LABEL_4:
      if ( (*(_BYTE *)(a1 + 32) & 0x10) == 0 )
        return result;
      goto LABEL_13;
    }
  }
  result = *(unsigned __int8 *)(a1 + 33);
  if ( (result & 2) == 0 )
    goto LABEL_3;
LABEL_9:
  v9 = sub_A748E0(v4, a3);
  if ( !v9 )
  {
    v10 = *(_QWORD *)(a2 - 32);
    if ( v10 )
    {
      if ( !*(_BYTE *)v10 && *(_QWORD *)(v10 + 24) == *(_QWORD *)(a2 + 80) )
      {
        v14[0] = *(_QWORD *)(v10 + 120);
        v9 = sub_A748E0(v14, a3);
      }
    }
  }
  *(_QWORD *)(a1 + 40) = v9;
  result = *(unsigned __int8 *)(a1 + 33);
  if ( (result & 1) == 0 )
    goto LABEL_4;
LABEL_11:
  result = sub_A74900(v4, a3);
  if ( !result )
  {
    v11 = *(_QWORD *)(a2 - 32);
    if ( v11 )
    {
      if ( !*(_BYTE *)v11 && *(_QWORD *)(v11 + 24) == *(_QWORD *)(a2 + 80) )
      {
        v14[0] = *(_QWORD *)(v11 + 120);
        result = sub_A74900(v14, a3);
      }
    }
  }
  *(_QWORD *)(a1 + 40) = result;
  if ( (*(_BYTE *)(a1 + 32) & 0x10) != 0 )
  {
LABEL_13:
    result = sub_A748C0(v4, a3);
    if ( !result )
    {
      v12 = *(_QWORD *)(a2 - 32);
      if ( v12 )
      {
        if ( !*(_BYTE *)v12 && *(_QWORD *)(v12 + 24) == *(_QWORD *)(a2 + 80) )
        {
          v14[0] = *(_QWORD *)(v12 + 120);
          result = sub_A748C0(v14, a3);
        }
      }
    }
    *(_QWORD *)(a1 + 40) = result;
  }
  return result;
}
