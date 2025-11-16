// Function: sub_120CD10
// Address: 0x120cd10
//
__int64 __fastcall sub_120CD10(__int64 a1, _BYTE *a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  int v6; // eax
  unsigned __int64 v7; // r15
  const char *v8; // rax
  unsigned __int64 v9; // rax
  char v10; // dl
  unsigned __int64 v11; // [rsp-70h] [rbp-70h] BYREF
  _QWORD v12[4]; // [rsp-68h] [rbp-68h] BYREF
  char v13; // [rsp-48h] [rbp-48h]
  char v14; // [rsp-47h] [rbp-47h]

  a2[1] = 0;
  result = 0;
  if ( *(_DWORD *)(a1 + 240) == 251 )
  {
    v4 = a1 + 176;
    v6 = sub_1205200(a1 + 176);
    v11 = 0;
    v7 = *(_QWORD *)(a1 + 232);
    *(_DWORD *)(a1 + 240) = v6;
    if ( (_BYTE)a3 && v6 == 12 )
    {
      *(_DWORD *)(a1 + 240) = sub_1205200(v4);
      if ( !(unsigned __int8)sub_120C050(a1, (__int64 *)&v11) )
      {
        if ( *(_DWORD *)(a1 + 240) != 13 )
        {
          v14 = 1;
          v12[0] = "expected ')'";
          v13 = 3;
          sub_11FD800(v4, v7, (__int64)v12, 1);
          return a3;
        }
        *(_DWORD *)(a1 + 240) = sub_1205200(v4);
        goto LABEL_4;
      }
    }
    else if ( !(unsigned __int8)sub_120C050(a1, (__int64 *)&v11) )
    {
LABEL_4:
      if ( !v11 || (v11 & (v11 - 1)) != 0 )
      {
        v14 = 1;
        v8 = "alignment is not a power of two";
      }
      else
      {
        if ( v11 <= 0x100000000LL )
        {
          _BitScanReverse64(&v9, v11);
          a2[1] = 1;
          v10 = 63 - (v9 ^ 0x3F);
          *a2 = v10;
          return 0;
        }
        v14 = 1;
        v8 = "huge alignments are not supported yet";
      }
      v12[0] = v8;
      v13 = 3;
      sub_11FD800(v4, v7, (__int64)v12, 1);
    }
    return 1;
  }
  return result;
}
