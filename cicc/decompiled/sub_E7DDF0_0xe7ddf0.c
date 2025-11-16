// Function: sub_E7DDF0
// Address: 0xe7ddf0
//
__int64 __fastcall sub_E7DDF0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r13d
  __int64 v10; // rsi
  unsigned int v11; // eax
  int v12; // edx
  unsigned int v13; // ecx
  char *v14; // rax
  unsigned int v15; // eax
  int v16; // edx
  __int64 v17; // rsi
  char *v18; // rax
  unsigned int v19; // eax
  int v20; // edx
  char *v21; // rax
  unsigned int v22; // eax
  int v23; // edx
  char *v24; // rax
  unsigned int v25; // eax
  int v26; // edx
  char *v27; // rax
  unsigned int v28; // eax
  int v29; // edx
  __int64 v30; // rsi
  char *v31; // rax
  __int64 v32; // rdi
  __int64 *v33; // rax
  __int64 v34; // rdx
  _QWORD *v35; // rax
  _QWORD *v36; // rsi
  __int64 v37; // rdi
  __int64 *v38; // rax
  __int64 v39; // rdx
  _QWORD *v40; // rax
  _QWORD *v41; // rsi
  __int64 v42; // rdi
  __int64 *v43; // rax
  __int64 v44; // rdx
  _QWORD *v45; // rax
  _QWORD *v46; // rsi
  _QWORD *v47; // [rsp+0h] [rbp-50h] BYREF
  __int64 v48; // [rsp+8h] [rbp-48h]
  const char *v49; // [rsp+10h] [rbp-40h]
  __int16 v50; // [rsp+20h] [rbp-30h]

  v7 = a3;
  sub_E5CB20(a1[37], a2, a3, a4, a5, a6);
  switch ( v7 )
  {
    case 0u:
    case 1u:
    case 0xBu:
    case 0xDu:
    case 0xEu:
    case 0x10u:
    case 0x13u:
    case 0x15u:
    case 0x17u:
    case 0x19u:
    case 0x1Bu:
    case 0x1Cu:
      return 0;
    case 2u:
      v11 = sub_EA1630(a2);
      v12 = 0;
      v13 = v11;
      v14 = (char *)&unk_3F805B0;
      if ( !v13 )
        goto LABEL_12;
      while ( v12 != 2 )
      {
        v14 += 4;
        if ( v14 != "0x%02lx" )
        {
          v12 = *(_DWORD *)v14;
          if ( v13 != *(_DWORD *)v14 )
            continue;
        }
        goto LABEL_12;
      }
      v7 = v13;
LABEL_12:
      sub_EA15B0(a2, v7);
      return 1;
    case 3u:
      v15 = sub_EA1630(a2);
      v16 = 0;
      v17 = v15;
      v18 = (char *)&unk_3F805B0;
      if ( !(_DWORD)v17 )
        goto LABEL_17;
      while ( v16 != 10 )
      {
        v18 += 4;
        if ( "0x%02lx" != v18 )
        {
          v16 = *(_DWORD *)v18;
          if ( (_DWORD)v17 != *(_DWORD *)v18 )
            continue;
        }
LABEL_17:
        v17 = 10;
        break;
      }
      sub_EA15B0(a2, v17);
      *(_BYTE *)(sub_E7DDE0((__int64)a1) + 201) = 1;
      return 1;
    case 4u:
      v25 = sub_EA1630(a2);
      v26 = 0;
      v10 = v25;
      v27 = (char *)&unk_3F805B0;
      if ( !(_DWORD)v10 )
        goto LABEL_28;
      while ( v26 != 1 )
      {
        v27 += 4;
        if ( v27 != "0x%02lx" )
        {
          v26 = *(_DWORD *)v27;
          if ( (_DWORD)v10 != *(_DWORD *)v27 )
            continue;
        }
        goto LABEL_28;
      }
      goto LABEL_4;
    case 5u:
      v19 = sub_EA1630(a2);
      v20 = 0;
      v10 = v19;
      v21 = (char *)&unk_3F805B0;
      if ( !(_DWORD)v10 )
        goto LABEL_23;
      while ( v20 != 6 )
      {
        v21 += 4;
        if ( "0x%02lx" != v21 )
        {
          v20 = *(_DWORD *)v21;
          if ( (_DWORD)v10 != *(_DWORD *)v21 )
            continue;
        }
LABEL_23:
        v10 = 6;
        goto LABEL_4;
      }
      goto LABEL_4;
    case 6u:
      v22 = sub_EA1630(a2);
      v23 = 0;
      v10 = v22;
      v24 = (char *)&unk_3F805B0;
      if ( !(_DWORD)v10 )
        goto LABEL_28;
      while ( v23 != 1 )
      {
        v24 += 4;
        if ( v24 != "0x%02lx" )
        {
          v23 = *(_DWORD *)v24;
          if ( (_DWORD)v10 != *(_DWORD *)v24 )
            continue;
        }
LABEL_28:
        v10 = 1;
        break;
      }
LABEL_4:
      sub_EA15B0(a2, v10);
      return 1;
    case 7u:
      v10 = (unsigned int)sub_EA1630(a2);
      goto LABEL_4;
    case 8u:
      v28 = sub_EA1630(a2);
      v29 = 0;
      v30 = v28;
      v31 = (char *)&unk_3F805B0;
      if ( !(_DWORD)v30 )
        goto LABEL_38;
      break;
    case 9u:
      if ( (unsigned __int8)sub_EA1770(a2) && (unsigned int)sub_EA1780(a2) != 1 )
      {
        v37 = a1[1];
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v38 = *(__int64 **)(a2 - 8);
          v39 = *v38;
          v40 = v38 + 3;
        }
        else
        {
          v39 = 0;
          v40 = 0;
        }
        v41 = (_QWORD *)a1[33];
        v47 = v40;
        v50 = 773;
        v48 = v39;
        v49 = " changed binding to STB_GLOBAL";
        if ( v41 )
          v41 = (_QWORD *)*v41;
        sub_E66880(v37, v41, (__int64)&v47);
      }
      sub_EA1710(a2, 1);
      return 1;
    case 0xAu:
    case 0x14u:
      BUG();
    case 0xCu:
      sub_EA1660(a2, 2);
      return 1;
    case 0xFu:
      sub_EA1660(a2, 1);
      return 1;
    case 0x11u:
      if ( (unsigned __int8)sub_EA1770(a2) && (unsigned int)sub_EA1780(a2) )
      {
        v32 = a1[1];
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v33 = *(__int64 **)(a2 - 8);
          v34 = *v33;
          v35 = v33 + 3;
        }
        else
        {
          v34 = 0;
          v35 = 0;
        }
        v36 = (_QWORD *)a1[33];
        v47 = v35;
        v50 = 773;
        v48 = v34;
        v49 = " changed binding to STB_LOCAL";
        if ( v36 )
          v36 = (_QWORD *)*v36;
        sub_E66880(v32, v36, (__int64)&v47);
      }
      sub_EA1710(a2, 0);
      return 1;
    case 0x16u:
      sub_EA1660(a2, 3);
      return 1;
    case 0x18u:
    case 0x1Au:
      if ( (unsigned __int8)sub_EA1770(a2) && (unsigned int)sub_EA1780(a2) != 2 )
      {
        v42 = a1[1];
        if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
        {
          v43 = *(__int64 **)(a2 - 8);
          v44 = *v43;
          v45 = v43 + 3;
        }
        else
        {
          v44 = 0;
          v45 = 0;
        }
        v47 = v45;
        v50 = 773;
        v46 = (_QWORD *)a1[33];
        v48 = v44;
        v49 = " changed binding to STB_WEAK";
        if ( v46 )
          v46 = (_QWORD *)*v46;
        sub_E668E0(v42, v46, (__int64)&v47);
      }
      sub_EA1710(a2, 2);
      return 1;
    case 0x1Du:
      sub_EA1850(a2, 1);
      return 1;
    default:
      return 1;
  }
  while ( v29 != 1 )
  {
    v31 += 4;
    if ( v31 != "0x%02lx" )
    {
      v29 = *(_DWORD *)v31;
      if ( (_DWORD)v30 != *(_DWORD *)v31 )
        continue;
    }
LABEL_38:
    v30 = 1;
    break;
  }
  sub_EA15B0(a2, v30);
  sub_EA1710(a2, 10);
  *(_BYTE *)(sub_E7DDE0((__int64)a1) + 201) = 1;
  return 1;
}
