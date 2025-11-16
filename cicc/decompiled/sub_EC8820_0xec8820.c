// Function: sub_EC8820
// Address: 0xec8820
//
__int64 __fastcall sub_EC8820(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r15
  const char *v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // r10d
  __int64 v14; // rdi
  unsigned int v15; // r15d
  __int64 v16; // rdi
  _DWORD *v17; // rdi
  __int64 v18; // rdi
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rdi
  void (*v23)(); // rax
  unsigned int v24; // [rsp+14h] [rbp-8Ch] BYREF
  unsigned int v25; // [rsp+18h] [rbp-88h] BYREF
  unsigned int v26; // [rsp+1Ch] [rbp-84h] BYREF
  __int64 v27; // [rsp+20h] [rbp-80h] BYREF
  __int64 v28; // [rsp+28h] [rbp-78h]
  __int128 v29; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v30[4]; // [rsp+40h] [rbp-60h] BYREF
  char v31; // [rsp+60h] [rbp-40h]
  char v32; // [rsp+61h] [rbp-3Fh]

  v7 = *(_QWORD *)(a1 + 8);
  v27 = 0;
  v28 = 0;
  v8 = sub_ECD7B0(v7);
  v9 = sub_ECD6A0(v8);
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64 *))(**(_QWORD **)(a1 + 8) + 192LL))(*(_QWORD *)(a1 + 8), &v27) )
  {
    v32 = 1;
    v10 = "platform name expected";
LABEL_3:
    v11 = *(_QWORD *)(a1 + 8);
    v30[0] = v10;
    v31 = 3;
    return (unsigned int)sub_ECE0E0(v11, v30, 0, 0);
  }
  if ( v28 != 7 )
  {
    switch ( v28 )
    {
      case 5LL:
        if ( *(_DWORD *)v27 == 1868783981 && *(_BYTE *)(v27 + 4) == 115 )
        {
          v15 = 1;
          goto LABEL_21;
        }
        break;
      case 3LL:
        if ( *(_WORD *)v27 == 28521 && *(_BYTE *)(v27 + 2) == 115 )
        {
          v15 = 2;
          goto LABEL_21;
        }
        break;
      case 4LL:
        if ( *(_DWORD *)v27 == 1936684660 )
        {
          v15 = 3;
          goto LABEL_21;
        }
        if ( *(_DWORD *)v27 == 1936683640 )
        {
          v15 = 11;
          goto LABEL_21;
        }
        break;
      case 8LL:
        if ( *(_QWORD *)v27 == 0x736F656764697262LL )
        {
          v15 = 5;
          goto LABEL_21;
        }
        break;
      case 11LL:
        if ( *(_QWORD *)v27 == 0x6C6174614363616DLL && *(_WORD *)(v27 + 8) == 29561 && *(_BYTE *)(v27 + 10) == 116 )
        {
          v15 = 6;
          goto LABEL_21;
        }
        if ( *(_QWORD *)v27 == 0x616C756D69737278LL && *(_WORD *)(v27 + 8) == 28532 && *(_BYTE *)(v27 + 10) == 114 )
        {
          v15 = 12;
          goto LABEL_21;
        }
        break;
      case 12LL:
        if ( *(_QWORD *)v27 == 0x6C756D6973736F69LL && *(_DWORD *)(v27 + 8) == 1919906913 )
        {
          v15 = 7;
          goto LABEL_21;
        }
        break;
      case 13LL:
        if ( *(_QWORD *)v27 == 0x756D6973736F7674LL && *(_DWORD *)(v27 + 8) == 1869898092 && *(_BYTE *)(v27 + 12) == 114 )
        {
          v15 = 8;
          goto LABEL_21;
        }
        break;
      case 16LL:
        if ( !(*(_QWORD *)v27 ^ 0x73736F6863746177LL | *(_QWORD *)(v27 + 8) ^ 0x726F74616C756D69LL) )
        {
          v15 = 9;
          goto LABEL_21;
        }
        break;
      default:
        if ( v28 == 9 && *(_QWORD *)v27 == 0x696B726576697264LL && *(_BYTE *)(v27 + 8) == 116 )
        {
          v15 = 10;
          goto LABEL_21;
        }
        break;
    }
LABEL_8:
    v14 = *(_QWORD *)(a1 + 8);
    v32 = 1;
    v30[0] = "unknown platform name";
    v31 = 3;
    return (unsigned int)sub_ECDA70(v14, v9, v30, 0, 0);
  }
  if ( *(_DWORD *)v27 == 1852534389 && *(_WORD *)(v27 + 4) == 30575 && *(_BYTE *)(v27 + 6) == 110
    || *(_DWORD *)v27 != 1668571511
    || *(_WORD *)(v27 + 4) != 28520
    || *(_BYTE *)(v27 + 6) != 115 )
  {
    goto LABEL_8;
  }
  v15 = 4;
LABEL_21:
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD, __int64 *, __int64))(**(_QWORD **)(a1 + 8) + 40LL))(
                       *(_QWORD *)(a1 + 8),
                       &v27,
                       v27)
                   + 8) != 26 )
  {
    v32 = 1;
    v10 = "version number required, comma expected";
    goto LABEL_3;
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  if ( (unsigned __int8)sub_EC83C0(a1, &v24, &v25, "OS") )
    return 1;
  if ( (unsigned __int8)sub_EC73D0(a1, &v26) )
    return 1;
  v16 = *(_QWORD *)(a1 + 8);
  v29 = 0;
  v17 = *(_DWORD **)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v16 + 40LL))(v16) + 8);
  if ( *v17 == 2 && sub_EC5140((__int64)v17) && (unsigned __int8)sub_EC8740(a1, (__int64)&v29) )
  {
    return 1;
  }
  else if ( (unsigned __int8)sub_ECE000(*(_QWORD *)(a1 + 8)) )
  {
    v18 = *(_QWORD *)(a1 + 8);
    v32 = 1;
    v30[0] = " in '.build_version' directive";
    v31 = 3;
    return (unsigned int)sub_ECD7F0(v18, v30);
  }
  else
  {
    switch ( v15 )
    {
      case 1u:
        v19 = 9;
        break;
      case 2u:
      case 6u:
        v19 = 5;
        break;
      case 3u:
        v19 = 27;
        break;
      case 4u:
        v19 = 28;
        break;
      case 0xAu:
        v19 = 30;
        break;
      case 0xBu:
        v19 = 31;
        break;
      default:
        BUG();
    }
    sub_EC6AF0(a1, a2, a3, v27, v28, a4, v19);
    v20 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v12 = 0;
    v22 = v20;
    v23 = *(void (**)())(*(_QWORD *)v20 + 256LL);
    if ( v23 != nullsub_103 )
    {
      ((void (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, __int64, _QWORD, _QWORD))v23)(
        v22,
        v15,
        v24,
        v25,
        v26,
        v21,
        v29,
        *((_QWORD *)&v29 + 1));
      return 0;
    }
  }
  return v12;
}
