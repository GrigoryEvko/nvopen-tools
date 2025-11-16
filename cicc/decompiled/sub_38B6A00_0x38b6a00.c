// Function: sub_38B6A00
// Address: 0x38b6a00
//
__int64 __fastcall sub_38B6A00(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  int v4; // eax
  const char *v5; // rax
  unsigned __int64 v6; // rsi
  int v7; // eax
  char v8; // al
  int *v9; // [rsp+0h] [rbp-90h]
  char v10; // [rsp+0h] [rbp-90h]
  int *v11; // [rsp+0h] [rbp-90h]
  int *v12; // [rsp+0h] [rbp-90h]
  unsigned __int8 v13; // [rsp+8h] [rbp-88h]
  int *v14; // [rsp+8h] [rbp-88h]
  int *v15; // [rsp+10h] [rbp-80h] BYREF
  __int64 v16; // [rsp+18h] [rbp-78h] BYREF
  _BYTE *v17; // [rsp+20h] [rbp-70h] BYREF
  __int64 v18; // [rsp+28h] [rbp-68h]
  _BYTE v19[16]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v20[2]; // [rsp+40h] [rbp-50h] BYREF
  char v21; // [rsp+50h] [rbp-40h] BYREF
  char v22; // [rsp+51h] [rbp-3Fh]

  v2 = a1 + 8;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  v4 = *(_DWORD *)(a1 + 64);
  v19[0] = 0;
  v17 = v19;
  v18 = 0;
  v15 = 0;
  if ( v4 == 306 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(v2);
    if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") || (unsigned __int8)sub_388BD80(a1, (__int64 *)&v15) )
      goto LABEL_15;
  }
  else
  {
    if ( v4 != 307 )
    {
      v22 = 1;
      v5 = "expected name or guid tag";
LABEL_8:
      v6 = *(_QWORD *)(a1 + 56);
      v20[0] = (__int64)v5;
      v21 = 3;
      result = sub_38814C0(v2, v6, (__int64)v20);
      goto LABEL_9;
    }
    *(_DWORD *)(a1 + 64) = sub_3887100(v2);
    if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
      || (unsigned __int8)sub_388B0A0(a1, (unsigned __int64 *)&v17) )
    {
      goto LABEL_15;
    }
  }
  if ( *(_DWORD *)(a1 + 64) == 4 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(v2);
    if ( !(unsigned __int8)sub_388AF10(a1, 308, "expected 'summaries' here")
      && !(unsigned __int8)sub_388AF10(a1, 16, "expected ':' here") )
    {
      while ( !(unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
      {
        v7 = *(_DWORD *)(a1 + 64);
        switch ( v7 )
        {
          case 314:
            v12 = v15;
            sub_2241BD0(v20, (__int64)&v17);
            v8 = sub_38B6140(a1, (__int64)v20, v12, a2);
            break;
          case 328:
            v11 = v15;
            sub_2241BD0(v20, (__int64)&v17);
            v8 = sub_38B4C10(a1, (__int64)v20, v11, a2);
            break;
          case 91:
            v9 = v15;
            sub_2241BD0(v20, (__int64)&v17);
            v8 = sub_38961C0(a1, (__int64)v20, v9, a2);
            break;
          default:
            v22 = 1;
            v5 = "expected summary type";
            goto LABEL_8;
        }
        v10 = v8;
        sub_2240A30((unsigned __int64 *)v20);
        if ( v10 || (unsigned __int8)sub_388AF10(a1, 13, "expected ')' here") )
          break;
        if ( *(_DWORD *)(a1 + 64) != 4 )
        {
          result = sub_388AF10(a1, 13, "expected ')' here");
          goto LABEL_9;
        }
        *(_DWORD *)(a1 + 64) = sub_3887100(v2);
      }
    }
  }
  else if ( !(unsigned __int8)sub_388AF10(a1, 13, "expected ')' here") )
  {
    v20[0] = (__int64)&v21;
    v14 = v15;
    v16 = 0;
    sub_3887850(v20, v17, (__int64)&v17[v18]);
    sub_3895460(a1, (__int64)v20, v14, 0, a2, &v16);
    result = 0;
    if ( (char *)v20[0] != &v21 )
    {
      j_j___libc_free_0(v20[0]);
      result = 0;
    }
    if ( v16 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
      result = 0;
    }
    goto LABEL_9;
  }
LABEL_15:
  result = 1;
LABEL_9:
  if ( v17 != v19 )
  {
    v13 = result;
    j_j___libc_free_0((unsigned __int64)v17);
    return v13;
  }
  return result;
}
