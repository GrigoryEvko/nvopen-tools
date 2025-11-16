// Function: sub_120B060
// Address: 0x120b060
//
__int64 __fastcall sub_120B060(__int64 a1, __int64 *a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // eax
  __int64 v8; // r14
  unsigned int v9; // r15d
  __int64 v10; // r15
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  int v13; // eax
  unsigned int v14; // eax
  __int64 v15; // r15
  const char *v16; // rax
  unsigned __int64 v17; // rsi
  __int64 *v19; // rdi
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // r8
  __int64 v23; // rdx
  unsigned __int64 v24; // rcx
  unsigned int v25; // eax
  int v26; // [rsp+14h] [rbp-ECh]
  _QWORD v28[4]; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v29; // [rsp+40h] [rbp-C0h]
  _QWORD v30[2]; // [rsp+50h] [rbp-B0h] BYREF
  char *v31; // [rsp+60h] [rbp-A0h]
  __int16 v32; // [rsp+70h] [rbp-90h]
  __int64 *v33; // [rsp+80h] [rbp-80h] BYREF
  __int64 v34; // [rsp+88h] [rbp-78h]
  _BYTE v35[112]; // [rsp+90h] [rbp-70h] BYREF

  v34 = 0x800000000LL;
  v7 = *(_DWORD *)(a1 + 240);
  v33 = (__int64 *)v35;
  if ( v7 == 13 )
    goto LABEL_20;
  v8 = a1 + 176;
  if ( v7 == 520 )
    goto LABEL_13;
LABEL_3:
  if ( v7 == 514 )
  {
    v25 = sub_E09EB0(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256));
    v15 = v25;
    if ( !v25 )
    {
      v16 = "invalid DWARF attribute encoding '";
LABEL_15:
      v28[0] = v16;
      v28[2] = a1 + 248;
      v30[0] = v28;
      v29 = 1027;
      v31 = "'";
      v32 = 770;
      goto LABEL_16;
    }
    while ( 1 )
    {
      v21 = sub_1205200(v8);
      v23 = (unsigned int)v34;
      v24 = HIDWORD(v34);
      *(_DWORD *)(a1 + 240) = v21;
      if ( v23 + 1 > v24 )
      {
        sub_C8D5F0((__int64)&v33, v35, v23 + 1, 8u, v22, v23 + 1);
        v23 = (unsigned int)v34;
      }
      v33[v23] = v15;
      v13 = *(_DWORD *)(a1 + 240);
      LODWORD(v34) = v34 + 1;
LABEL_11:
      if ( v13 != 4 )
        break;
      v7 = sub_1205200(v8);
      *(_DWORD *)(a1 + 240) = v7;
      if ( v7 != 520 )
        goto LABEL_3;
LABEL_13:
      v14 = sub_E07820(*(_QWORD *)(a1 + 248), *(_QWORD *)(a1 + 256));
      v15 = v14;
      if ( !v14 )
      {
        v16 = "invalid DWARF op '";
        goto LABEL_15;
      }
    }
LABEL_20:
    v17 = 13;
    v9 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( !(_BYTE)v9 )
    {
      v17 = (unsigned __int64)v33;
      v19 = *(__int64 **)a1;
      if ( a3 )
        v20 = sub_B0D000(v19, v33, (unsigned int)v34, 1u, 1);
      else
        v20 = sub_B0D000(v19, v33, (unsigned int)v34, 0, 1);
      *a2 = v20;
    }
    goto LABEL_17;
  }
  if ( v7 != 529 || (v9 = *(unsigned __int8 *)(a1 + 332), !(_BYTE)v9) )
  {
    v30[0] = "expected unsigned integer";
    v32 = 259;
LABEL_16:
    v17 = *(_QWORD *)(a1 + 232);
    v9 = 1;
    sub_11FD800(v8, v17, (__int64)v30, 1);
    goto LABEL_17;
  }
  if ( *(_DWORD *)(a1 + 328) <= 0x40u )
  {
    v10 = *(_QWORD *)(a1 + 320);
LABEL_8:
    v11 = (unsigned int)v34;
    v12 = (unsigned int)v34 + 1LL;
    if ( v12 > HIDWORD(v34) )
    {
      sub_C8D5F0((__int64)&v33, v35, v12, 8u, a5, a6);
      v11 = (unsigned int)v34;
    }
    v33[v11] = v10;
    LODWORD(v34) = v34 + 1;
    v13 = sub_1205200(v8);
    *(_DWORD *)(a1 + 240) = v13;
    goto LABEL_11;
  }
  v26 = *(_DWORD *)(a1 + 328);
  if ( v26 - (unsigned int)sub_C444A0(a1 + 320) <= 0x40 )
  {
    v10 = **(_QWORD **)(a1 + 320);
    goto LABEL_8;
  }
  v17 = *(_QWORD *)(a1 + 232);
  v30[0] = "element too large, limit is ";
  v31 = (char *)v28;
  v28[0] = -1;
  v32 = 2819;
  sub_11FD800(v8, v17, (__int64)v30, 1);
LABEL_17:
  if ( v33 != (__int64 *)v35 )
    _libc_free(v33, v17);
  return v9;
}
