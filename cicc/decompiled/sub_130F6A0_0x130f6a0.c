// Function: sub_130F6A0
// Address: 0x130f6a0
//
__int64 __fastcall sub_130F6A0(__int64 a1, __int64 a2, char *a3)
{
  int v3; // edx
  int v4; // ecx
  int v5; // r8d
  int v6; // r9d
  char *v7; // rdx
  int v9; // eax
  __int64 result; // rax
  char v11; // al
  __int64 v12; // rdx
  int v13; // r14d
  char v14; // cl
  char v15; // r15
  char v16; // [rsp+0h] [rbp-70h]
  unsigned __int8 v17; // [rsp+3h] [rbp-6Dh]
  char v18; // [rsp+4h] [rbp-6Ch]
  char v19; // [rsp+8h] [rbp-68h]
  unsigned __int8 v20; // [rsp+Ch] [rbp-64h]
  unsigned __int8 v21; // [rsp+Dh] [rbp-63h]
  unsigned __int8 v22; // [rsp+Eh] [rbp-62h]
  unsigned __int8 v23; // [rsp+Fh] [rbp-61h]
  __int64 v24; // [rsp+10h] [rbp-60h] BYREF
  __int64 v25; // [rsp+18h] [rbp-58h] BYREF
  unsigned int v26; // [rsp+20h] [rbp-50h] BYREF
  __int64 v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h]
  int v29; // [rsp+38h] [rbp-38h]
  __int16 v30; // [rsp+3Ch] [rbp-34h]

  v24 = 1;
  v25 = 8;
  v9 = sub_1308610((__int64)"epoch", (__int64)&v24, (__int64)&v25, (__int64)&v24, 8);
  if ( v9 )
  {
    if ( v9 != 11 )
    {
      sub_130AA40("<jemalloc>: Failure in mallctl(\"epoch\", ...)\n");
      abort();
    }
    return sub_130AA40("<jemalloc>: Memory allocation failure in mallctl(\"epoch\", ...)\n");
  }
  if ( !a3 || (v11 = *a3) == 0 )
  {
    v17 = 1;
    v13 = 1;
    v15 = 1;
    v20 = 1;
    v18 = 1;
    v19 = 1;
    v21 = 1;
    v22 = 1;
    v23 = 1;
LABEL_19:
    v28 = a2;
    v27 = a1;
    v26 = 2;
    v29 = 0;
    v30 = 0;
    sub_130F0B0((__int64)&v26, "%s", byte_3F871B3);
    goto LABEL_13;
  }
  v17 = 1;
  LODWORD(v12) = 0;
  v13 = 1;
  v14 = 0;
  v20 = 1;
  v15 = 1;
  v18 = 1;
  v19 = 1;
  v21 = 1;
  v22 = 1;
  v23 = 1;
  do
  {
    switch ( v11 )
    {
      case 'J':
        v14 = 1;
        break;
      case 'a':
        v21 = 0;
        break;
      case 'b':
        v19 = 0;
        break;
      case 'd':
        v22 = 0;
        break;
      case 'e':
        v17 = 0;
        break;
      case 'g':
        v15 = 0;
        break;
      case 'h':
        v13 = 0;
        break;
      case 'l':
        v18 = 0;
        break;
      case 'm':
        v23 = 0;
        break;
      case 'x':
        v20 = 0;
        break;
      default:
        break;
    }
    v12 = (unsigned int)(v12 + 1);
    v11 = a3[v12];
  }
  while ( v11 );
  if ( !v14 )
    goto LABEL_19;
  v28 = a2;
  v26 = 1;
  v27 = a1;
  v29 = 0;
  v30 = 0;
  sub_130F0B0((__int64)&v26, "{");
  ++v29;
  LOBYTE(v30) = 0;
LABEL_13:
  sub_130F1C0((__int64)&v26, "___ Begin jemalloc statistics ___\n");
  sub_130F560(&v26, "jemalloc");
  if ( v15 )
    sub_411419(&v26);
  sub_417CBD(&v26, v23, v22, v21, v19 & 1, v18 & 1, v20, v17, v13);
  sub_40E56D(&v26, v23, v3, v4, v5, v6, v16);
  sub_130F1C0((__int64)&v26, "--- End jemalloc statistics ---\n");
  result = v26;
  if ( v26 <= 1 )
  {
    --v29;
    v7 = "}";
    LOBYTE(v30) = 1;
    if ( v26 != 1 )
      v7 = "\n}\n";
    return sub_130F0B0((__int64)&v26, "%s", v7);
  }
  return result;
}
