// Function: sub_2DE6DC0
// Address: 0x2de6dc0
//
__int64 __fastcall sub_2DE6DC0(__int64 a1, unsigned int a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  unsigned int v7; // r13d
  bool v8; // al
  __int64 result; // rax
  int v10; // r13d
  char v11; // r14
  unsigned int v12; // r15d
  unsigned __int8 *v13; // rax
  unsigned int v14; // r14d
  __int64 v15; // rax
  unsigned int v16; // r13d
  bool v17; // r14
  unsigned __int8 *v18; // rax
  __int64 v19; // r13
  __int64 v20; // rdx
  _BYTE *v21; // rax
  unsigned int v22; // r13d
  __int64 v23; // rax
  __int64 v24; // rcx
  _BYTE *v25; // r15
  __int64 v26; // [rsp+8h] [rbp-D8h]
  __int64 v27; // [rsp+8h] [rbp-D8h]
  void *s2; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v29; // [rsp+18h] [rbp-C8h]
  _BYTE v30[64]; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v31[2]; // [rsp+60h] [rbp-80h] BYREF
  _BYTE v32[112]; // [rsp+70h] [rbp-70h] BYREF

  v6 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 != 85 )
    goto LABEL_2;
  v23 = *(_QWORD *)(a1 - 32);
  if ( !v23 )
    return 0;
  if ( *(_BYTE *)v23 )
    return 0;
  v24 = *(_QWORD *)(a1 + 80);
  if ( *(_QWORD *)(v23 + 24) != v24 || (*(_BYTE *)(v23 + 33) & 0x20) == 0 )
    return 0;
  v29 = 0x800000000LL;
  s2 = v30;
  v31[0] = (__int64)v32;
  v31[1] = 0x800000000LL;
  if ( !(unsigned __int8)sub_2DE6B60(a1, (__int64)&s2, (__int64)v31, v24, a5, a6)
    || (_DWORD)v29 != a2
    || (v25 = s2, 8LL * a2) && 8LL * a2 != 8 && memcmp((char *)s2 + 8, s2, 8LL * a2 - 8) )
  {
    if ( (_BYTE *)v31[0] != v32 )
      _libc_free(v31[0]);
    if ( s2 != v30 )
      _libc_free((unsigned __int64)s2);
    v6 = *(_BYTE *)a1;
LABEL_2:
    if ( v6 == 17 )
    {
      v7 = *(_DWORD *)(a1 + 32);
      if ( !v7 )
        goto LABEL_21;
      if ( v7 <= 0x40 )
        v8 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v7) == *(_QWORD *)(a1 + 24);
      else
        v8 = v7 == (unsigned int)sub_C445E0(a1 + 24);
    }
    else
    {
      v19 = *(_QWORD *)(a1 + 8);
      v20 = (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17;
      if ( (unsigned int)v20 > 1 || v6 > 0x15u )
        return 0;
      v21 = sub_AD7630(a1, 0, v20);
      if ( !v21 || *v21 != 17 )
      {
        if ( *(_BYTE *)(v19 + 8) == 17 )
        {
          v10 = *(_DWORD *)(v19 + 32);
          if ( v10 )
          {
            v11 = 0;
            v12 = 0;
            while ( 1 )
            {
              v13 = (unsigned __int8 *)sub_AD69F0((unsigned __int8 *)a1, v12);
              if ( !v13 )
                break;
              a3 = *v13;
              if ( (_BYTE)a3 != 13 )
              {
                if ( (_BYTE)a3 != 17 )
                  return 0;
                v14 = *((_DWORD *)v13 + 8);
                if ( v14 )
                {
                  if ( v14 <= 0x40 )
                  {
                    a3 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v14);
                    if ( *((_QWORD *)v13 + 3) != a3 )
                      return 0;
                  }
                  else if ( v14 != (unsigned int)sub_C445E0((__int64)(v13 + 24)) )
                  {
                    return 0;
                  }
                }
                v11 = 1;
              }
              if ( v10 == ++v12 )
              {
                if ( !v11 )
                  return 0;
                goto LABEL_21;
              }
            }
          }
        }
        return 0;
      }
      v22 = *((_DWORD *)v21 + 8);
      if ( !v22 )
      {
LABEL_21:
        v15 = *(_QWORD *)(a1 + 8);
        v16 = *(_DWORD *)(v15 + 32);
        v17 = *(_BYTE *)(v15 + 8) == 18;
        v18 = sub_AD7630(a1, 0, a3);
        BYTE4(v31[0]) = v17;
        LODWORD(v31[0]) = v16 / a2;
        return sub_AD5E10(v31[0], v18);
      }
      if ( v22 <= 0x40 )
      {
        a3 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22);
        v8 = a3 == *((_QWORD *)v21 + 3);
      }
      else
      {
        v8 = v22 == (unsigned int)sub_C445E0((__int64)(v21 + 24));
      }
    }
    if ( !v8 )
      return 0;
    goto LABEL_21;
  }
  result = *(_QWORD *)v25;
  if ( (_BYTE *)v31[0] != v32 )
  {
    v26 = *(_QWORD *)v25;
    _libc_free(v31[0]);
    v25 = s2;
    result = v26;
  }
  if ( v25 != v30 )
  {
    v27 = result;
    _libc_free((unsigned __int64)v25);
    return v27;
  }
  return result;
}
