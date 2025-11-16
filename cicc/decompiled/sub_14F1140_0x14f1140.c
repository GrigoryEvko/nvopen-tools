// Function: sub_14F1140
// Address: 0x14f1140
//
__int64 *__fastcall sub_14F1140(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  const char *v3; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  const char *v7; // rax
  const char *v8; // [rsp+0h] [rbp-260h] BYREF
  char v9; // [rsp+10h] [rbp-250h]
  char v10; // [rsp+11h] [rbp-24Fh]
  __int64 *v11; // [rsp+20h] [rbp-240h] BYREF
  __int64 v12; // [rsp+28h] [rbp-238h]
  char v13; // [rsp+30h] [rbp-230h] BYREF
  char v14; // [rsp+31h] [rbp-22Fh]

  v2 = a2 + 32;
  if ( (unsigned __int8)sub_15127D0(a2 + 32, 21, 0) )
  {
    v14 = 1;
    v3 = "Invalid record";
    goto LABEL_4;
  }
  if ( *(_QWORD *)(a2 + 1744) != *(_QWORD *)(a2 + 1736) )
  {
    v14 = 1;
    v3 = "Invalid multiple blocks";
LABEL_4:
    v11 = (__int64 *)v3;
    v13 = 3;
    sub_14EE4B0(a1, a2 + 8, (__int64)&v11);
    return a1;
  }
  v11 = (__int64 *)&v13;
  v12 = 0x4000000000LL;
  v5 = sub_14ED070(v2, 0);
  if ( (_DWORD)v5 == 1 )
  {
LABEL_15:
    *a1 = 1;
  }
  else
  {
    while ( 1 )
    {
      if ( (v5 & 0xFFFFFFFD) == 0 )
      {
        v10 = 1;
        v7 = "Malformed block";
        goto LABEL_18;
      }
      if ( (unsigned int)sub_1510D70(v2, HIDWORD(v5), &v11, 0) != 1 )
        break;
      v6 = *(_QWORD *)(a2 + 1744);
      if ( v6 == *(_QWORD *)(a2 + 1752) )
      {
        sub_9CBC60((const __m128i **)(a2 + 1736), *(const __m128i **)(a2 + 1744));
        v6 = *(_QWORD *)(a2 + 1744) - 32LL;
      }
      else
      {
        if ( v6 )
        {
          *(_QWORD *)(v6 + 8) = 0;
          *(_QWORD *)v6 = v6 + 16;
          *(_BYTE *)(v6 + 16) = 0;
          v6 = *(_QWORD *)(a2 + 1744);
        }
        *(_QWORD *)(a2 + 1744) = v6 + 32;
      }
      if ( (unsigned __int8)sub_14EA4D0(v11, v12, (_QWORD *)v6) )
        break;
      LODWORD(v12) = 0;
      v5 = sub_14ED070(v2, 0);
      if ( (_DWORD)v5 == 1 )
        goto LABEL_15;
    }
    v10 = 1;
    v7 = "Invalid record";
LABEL_18:
    v8 = v7;
    v9 = 3;
    sub_14EE4B0(a1, a2 + 8, (__int64)&v8);
  }
  if ( v11 != (__int64 *)&v13 )
    _libc_free((unsigned __int64)v11);
  return a1;
}
