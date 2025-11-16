// Function: sub_2335540
// Address: 0x2335540
//
__int64 __fastcall sub_2335540(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  char v3; // r15
  char v4; // r14
  __int64 v5; // r13
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  _WORD *v8; // rcx
  unsigned __int64 v9; // r13
  __int64 v10; // rdx
  char v11; // r9
  char v12; // al
  unsigned int v14; // eax
  unsigned int v15; // ebx
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // rax
  char v19; // [rsp+8h] [rbp-D8h]
  char v20; // [rsp+9h] [rbp-D7h]
  __int16 v21; // [rsp+Ah] [rbp-D6h]
  __int16 v22; // [rsp+Ch] [rbp-D4h]
  char v23; // [rsp+Eh] [rbp-D2h]
  char v24; // [rsp+Fh] [rbp-D1h]
  _WORD *v25; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int64 v26; // [rsp+18h] [rbp-C8h]
  __int64 v27; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v28; // [rsp+32h] [rbp-AEh]
  int v29; // [rsp+3Ah] [rbp-A6h]
  __int16 v30; // [rsp+3Eh] [rbp-A2h]
  const void *v31; // [rsp+40h] [rbp-A0h] BYREF
  size_t v32; // [rsp+48h] [rbp-98h]
  unsigned __int64 v33[4]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v34[4]; // [rsp+70h] [rbp-70h] BYREF
  char v35; // [rsp+90h] [rbp-50h]
  _QWORD v36[2]; // [rsp+98h] [rbp-48h] BYREF
  _QWORD *v37; // [rsp+A8h] [rbp-38h] BYREF

  v3 = 0;
  v4 = 0;
  v5 = a1;
  v25 = (_WORD *)a2;
  v26 = a3;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v21 = 0;
  v19 = 0;
  v20 = 0;
  v24 = 0;
  v23 = 0;
  v22 = 0;
  if ( a3 )
  {
    while ( 1 )
    {
      v31 = 0;
      v32 = 0;
      LOBYTE(v34[0]) = 59;
      v6 = sub_C931B0((__int64 *)&v25, v34, 1u, 0);
      if ( v6 == -1 )
      {
        v8 = v25;
        v6 = v26;
        v9 = 0;
        v10 = 0;
      }
      else
      {
        v7 = v6 + 1;
        v8 = v25;
        if ( v6 + 1 > v26 )
        {
          v7 = v26;
          v9 = 0;
        }
        else
        {
          v9 = v26 - v7;
        }
        v10 = (__int64)v25 + v7;
        if ( v6 > v26 )
          v6 = v26;
      }
      v31 = v8;
      v11 = 1;
      v32 = v6;
      v25 = (_WORD *)v10;
      v26 = v9;
      if ( v6 <= 2 )
        goto LABEL_8;
      if ( *v8 == 28526 && *((_BYTE *)v8 + 2) == 45 )
      {
        v6 -= 3LL;
        v8 = (_WORD *)((char *)v8 + 3);
        v11 = 0;
        v31 = v8;
        v32 = v6;
      }
      else
      {
        v11 = 1;
      }
      if ( v6 != 3 )
        break;
      if ( *v8 == 29296 && *((_BYTE *)v8 + 2) == 101 )
      {
        LOBYTE(v22) = v11;
        HIBYTE(v22) = 1;
        goto LABEL_13;
      }
LABEL_11:
      v24 = v11;
      if ( !sub_9691B0(v31, v32, "memoryssa", 9) )
      {
        v5 = a1;
        v14 = sub_C63BB0();
        v35 = 1;
        v15 = v14;
        v17 = v16;
        v34[1] = 33;
        v34[0] = "invalid GVN pass parameter '{0}' ";
        v34[2] = &v37;
        v34[3] = 1;
        v36[0] = &unk_49DB108;
        v36[1] = &v31;
        v37 = v36;
        sub_23328D0((__int64)v33, (__int64)v34);
        sub_23058C0(&v27, (__int64)v33, v15, v17);
        v18 = v27;
        *(_BYTE *)(a1 + 16) |= 3u;
        *(_QWORD *)a1 = v18 & 0xFFFFFFFFFFFFFFFELL;
        sub_2240A30(v33);
        return v5;
      }
      v23 = 1;
LABEL_13:
      if ( !v9 )
      {
        v5 = a1;
        goto LABEL_15;
      }
    }
    if ( v6 == 8 )
    {
      if ( *(_QWORD *)v8 == 0x6572702D64616F6CLL )
      {
        v3 = v11;
        v4 = 1;
        goto LABEL_13;
      }
      goto LABEL_11;
    }
LABEL_8:
    if ( v6 == 23 )
    {
      if ( !(*(_QWORD *)v8 ^ 0x61622D74696C7073LL | *((_QWORD *)v8 + 1) ^ 0x6C2D656764656B63LL)
        && *((_DWORD *)v8 + 4) == 761553263
        && v8[10] == 29296
        && *((_BYTE *)v8 + 22) == 101 )
      {
        v20 = v11;
        v19 = 1;
        goto LABEL_13;
      }
    }
    else if ( v6 == 6 && *(_DWORD *)v8 == 1684890989 && v8[2] == 28773 )
    {
      LOBYTE(v21) = v11;
      HIBYTE(v21) = 1;
      goto LABEL_13;
    }
    goto LABEL_11;
  }
LABEL_15:
  v12 = *(_BYTE *)(v5 + 16);
  BYTE4(v28) = v3;
  BYTE5(v28) = v4;
  *(_BYTE *)(v5 + 16) = v12 & 0xFC | 2;
  LOWORD(v28) = v22;
  LOBYTE(v29) = v20;
  BYTE1(v29) = v19;
  HIWORD(v29) = v21;
  LOBYTE(v30) = v24;
  HIBYTE(v30) = v23;
  *(_QWORD *)v5 = v28;
  *(_DWORD *)(v5 + 8) = v29;
  *(_WORD *)(v5 + 12) = v30;
  return v5;
}
