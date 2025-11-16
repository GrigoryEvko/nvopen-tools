// Function: sub_120D9C0
// Address: 0x120d9c0
//
const char *__fastcall sub_120D9C0(__int64 a1)
{
  char v1; // r13
  __int64 v2; // r12
  int v3; // eax
  unsigned __int64 v4; // rsi
  const char *result; // rax
  int v6; // r14d
  char v7; // r15
  unsigned int v8; // eax
  char v9; // cl
  char v10; // dl
  int v11; // eax
  int v12; // edx
  int v13; // esi
  int v14; // eax
  int v15; // eax
  const char *v16; // rax
  int v17; // eax
  char v18; // [rsp+Eh] [rbp-62h]
  unsigned __int8 v19; // [rsp+Fh] [rbp-61h]
  const char *v20; // [rsp+10h] [rbp-60h] BYREF
  char v21; // [rsp+30h] [rbp-40h]
  char v22; // [rsp+31h] [rbp-3Fh]

  v2 = a1 + 176;
  *(_BYTE *)(a1 + 336) = 1;
  v3 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v3;
  if ( v3 == 12 )
  {
    v6 = 0;
    v7 = 0;
    v8 = sub_1205200(v2);
    for ( *(_DWORD *)(a1 + 240) = v8; ; *(_DWORD *)(a1 + 240) = v8 )
    {
      if ( v8 == 268 )
      {
        v1 = 1;
        goto LABEL_21;
      }
      if ( v8 == 269 )
        break;
      if ( v8 == 267 )
      {
        v1 = 0;
        goto LABEL_21;
      }
      if ( v8 == 265 )
      {
        v15 = sub_1205200(v2);
        v12 = 2;
        *(_DWORD *)(a1 + 240) = v15;
        v13 = v15;
        goto LABEL_28;
      }
      v9 = 0;
LABEL_11:
      if ( v8 > 0x109 )
      {
        if ( v8 != 266 )
        {
LABEL_35:
          v22 = 1;
          v4 = *(_QWORD *)(a1 + 232);
          v16 = "expected memory location (argmem, inaccessiblemem, errnomem) or access kind (none, read, write, readwrite)";
          if ( v9 )
            v16 = "expected access kind (none, read, write, readwrite)";
          v20 = v16;
          v21 = 3;
          goto LABEL_3;
        }
        v10 = 3;
      }
      else if ( v8 == 55 )
      {
        v10 = 0;
      }
      else
      {
        if ( v8 != 264 )
          goto LABEL_35;
        v10 = 1;
      }
      v18 = v9;
      v19 = v10;
      v11 = sub_1205200(v2);
      v12 = v19;
      *(_DWORD *)(a1 + 240) = v11;
      v13 = v11;
      if ( v18 )
        goto LABEL_16;
LABEL_28:
      if ( v7 )
      {
        v22 = 1;
        v4 = *(_QWORD *)(a1 + 232);
        v20 = "default access kind must be specified first";
        v21 = 3;
        goto LABEL_3;
      }
      v6 = (v12 << 6) | (16 * v12) | v12 | (4 * v12);
LABEL_17:
      if ( v13 == 13 )
      {
        LODWORD(v20) = v6;
        *(_DWORD *)(a1 + 240) = sub_1205200(v2);
        BYTE4(v20) = 1;
        goto LABEL_4;
      }
      if ( v13 != 4 )
      {
        v22 = 1;
        v4 = *(_QWORD *)(a1 + 232);
        v20 = "unterminated memory attribute";
        v21 = 3;
        goto LABEL_3;
      }
      v8 = sub_1205200(v2);
    }
    v1 = 2;
LABEL_21:
    v14 = sub_1205200(v2);
    *(_DWORD *)(a1 + 240) = v14;
    if ( v14 != 16 )
    {
      v22 = 1;
      v4 = *(_QWORD *)(a1 + 232);
      v20 = "expected ':' after location";
      v21 = 3;
      goto LABEL_3;
    }
    v8 = sub_1205200(v2);
    *(_DWORD *)(a1 + 240) = v8;
    if ( v8 == 265 )
    {
      v17 = sub_1205200(v2);
      v12 = 2;
      *(_DWORD *)(a1 + 240) = v17;
      v13 = v17;
LABEL_16:
      v7 = 1;
      v6 = (v12 << (2 * v1)) | ~(3 << (2 * v1)) & v6;
      goto LABEL_17;
    }
    v9 = 1;
    goto LABEL_11;
  }
  v22 = 1;
  v4 = *(_QWORD *)(a1 + 232);
  v20 = "expected '('";
  v21 = 3;
LABEL_3:
  sub_11FD800(v2, v4, (__int64)&v20, 1);
  BYTE4(v20) = 0;
LABEL_4:
  result = v20;
  *(_BYTE *)(a1 + 336) = 0;
  return result;
}
