// Function: sub_1AC5CF0
// Address: 0x1ac5cf0
//
__int64 __fastcall sub_1AC5CF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rdx
  __int64 *v7; // rax
  unsigned int v8; // edx
  unsigned int v9; // r9d
  __int64 result; // rax
  int v11; // esi
  unsigned int v12; // edi
  unsigned __int16 v13; // si
  unsigned int v14; // r9d
  __int64 *v15; // rdi
  __int64 *v16; // rsi
  __int64 v17; // rax
  _QWORD *v18; // r12
  _QWORD *v19; // r13
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rax
  __int64 v23; // r9
  __int64 v24; // rsi
  __int64 v25; // rax
  unsigned int v26; // esi
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rax
  int v31; // eax
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // [rsp+8h] [rbp-58h]
  __int64 v37; // [rsp+10h] [rbp-50h]
  __int64 v38; // [rsp+10h] [rbp-50h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  unsigned __int64 v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41; // [rsp+20h] [rbp-40h]
  __int64 v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+20h] [rbp-40h]
  __int64 v46; // [rsp+28h] [rbp-38h]
  __int64 v47; // [rsp+28h] [rbp-38h]
  __int64 v48; // [rsp+28h] [rbp-38h]
  __int64 v49; // [rsp+28h] [rbp-38h]

  v6 = *(__int64 **)(a2 + 16);
  v7 = *(__int64 **)(a2 + 8);
  while ( 1 )
  {
    if ( v6 != v7 )
      goto LABEL_3;
    v15 = &v7[*(unsigned int *)(a2 + 28)];
    v9 = *(_DWORD *)(a2 + 28);
    if ( v15 != v7 )
    {
      v16 = 0;
      while ( a1 != *v7 )
      {
        if ( *v7 == -2 )
          v16 = v7;
        if ( v15 == ++v7 )
        {
          if ( !v16 )
            goto LABEL_29;
          *v16 = a1;
          v6 = *(__int64 **)(a2 + 16);
          --*(_DWORD *)(a2 + 32);
          v7 = *(__int64 **)(a2 + 8);
          ++*(_QWORD *)a2;
          goto LABEL_8;
        }
      }
      return 1;
    }
LABEL_29:
    if ( v9 < *(_DWORD *)(a2 + 24) )
    {
      *(_DWORD *)(a2 + 28) = ++v9;
      *v15 = a1;
      v7 = *(__int64 **)(a2 + 8);
      ++*(_QWORD *)a2;
      v6 = *(__int64 **)(a2 + 16);
    }
    else
    {
LABEL_3:
      sub_16CCBA0(a2, a1);
      v7 = *(__int64 **)(a2 + 8);
      v9 = v8;
      v6 = *(__int64 **)(a2 + 16);
      if ( !(_BYTE)v9 )
        return 1;
    }
LABEL_8:
    v11 = *(unsigned __int8 *)(a1 + 16);
    if ( (unsigned __int8)v11 <= 3u )
    {
      LOBYTE(v9) = (*(_BYTE *)(a1 + 33) & 0x1C) == 0;
      if ( (*(_BYTE *)(a1 + 33) & 3) != 1 )
        return v9;
      return 0;
    }
    v12 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    LOBYTE(v9) = v12 == 0 || (_BYTE)v11 == 4;
    if ( (_BYTE)v9 )
      return v9;
    if ( (unsigned int)(v11 - 6) <= 2 )
      break;
    v13 = *(_WORD *)(a1 + 18);
    if ( v13 > 0x2Eu )
    {
      if ( v13 != 47 )
        return v9;
      a1 = *(_QWORD *)(a1 - 24LL * v12);
    }
    else if ( v13 > 0x2Cu )
    {
      v20 = *(_QWORD *)a1;
      v21 = 1;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v20 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v28 = *(_QWORD *)(v20 + 32);
            v20 = *(_QWORD *)(v20 + 24);
            v21 *= v28;
            continue;
          case 1:
            v22 = 16;
            goto LABEL_44;
          case 2:
            v22 = 32;
            goto LABEL_44;
          case 3:
          case 9:
            v22 = 64;
            goto LABEL_44;
          case 4:
            v22 = 80;
            goto LABEL_44;
          case 5:
          case 6:
            v22 = 128;
            goto LABEL_44;
          case 7:
            v47 = v21;
            v26 = 0;
            goto LABEL_54;
          case 0xB:
            v22 = *(_DWORD *)(v20 + 8) >> 8;
            goto LABEL_44;
          case 0xD:
            v49 = v21;
            v30 = (_QWORD *)sub_15A9930(a3, v20);
            v21 = v49;
            v12 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
            v22 = 8LL * *v30;
            goto LABEL_44;
          case 0xE:
            v37 = v21;
            v39 = *(_QWORD *)(v20 + 24);
            v48 = *(_QWORD *)(v20 + 32);
            v41 = (unsigned int)sub_15A9FE0(a3, v39);
            v29 = sub_127FA20(a3, v39);
            v21 = v37;
            v12 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
            v22 = 8 * v48 * v41 * ((v41 + ((unsigned __int64)(v29 + 7) >> 3) - 1) / v41);
            goto LABEL_44;
          case 0xF:
            v47 = v21;
            v26 = *(_DWORD *)(v20 + 8) >> 8;
LABEL_54:
            v27 = sub_15A9520(a3, v26);
            v21 = v47;
            v22 = (unsigned int)(8 * v27);
            v12 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
LABEL_44:
            v23 = 1;
            v46 = v22 * v21;
            v24 = **(_QWORD **)(a1 - 24LL * v12);
            while ( 2 )
            {
              switch ( *(_BYTE *)(v24 + 8) )
              {
                case 1:
                  v25 = 16;
                  goto LABEL_48;
                case 2:
                  v25 = 32;
                  goto LABEL_48;
                case 3:
                case 9:
                  v25 = 64;
                  goto LABEL_48;
                case 4:
                  v25 = 80;
                  goto LABEL_48;
                case 5:
                case 6:
                  v25 = 128;
                  goto LABEL_48;
                case 7:
                  v42 = v23;
                  v31 = sub_15A9520(a3, 0);
                  v23 = v42;
                  v25 = (unsigned int)(8 * v31);
                  goto LABEL_48;
                case 0xB:
                  v25 = *(_DWORD *)(v24 + 8) >> 8;
                  goto LABEL_48;
                case 0xD:
                  v45 = v23;
                  v35 = (_QWORD *)sub_15A9930(a3, v24);
                  v23 = v45;
                  v25 = 8LL * *v35;
                  goto LABEL_48;
                case 0xE:
                  v36 = v23;
                  v38 = *(_QWORD *)(v24 + 24);
                  v44 = *(_QWORD *)(v24 + 32);
                  v40 = (unsigned int)sub_15A9FE0(a3, v38);
                  v34 = sub_127FA20(a3, v38);
                  v23 = v36;
                  v25 = 8 * v44 * v40 * ((v40 + ((unsigned __int64)(v34 + 7) >> 3) - 1) / v40);
                  goto LABEL_48;
                case 0xF:
                  v43 = v23;
                  v33 = sub_15A9520(a3, *(_DWORD *)(v24 + 8) >> 8);
                  v23 = v43;
                  v25 = (unsigned int)(8 * v33);
LABEL_48:
                  if ( v46 != v23 * v25 )
                    return 0;
                  v6 = *(__int64 **)(a2 + 16);
                  a1 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
                  v7 = *(__int64 **)(a2 + 8);
                  break;
                case 0x10:
                  v32 = *(_QWORD *)(v24 + 32);
                  v24 = *(_QWORD *)(v24 + 24);
                  v23 *= v32;
                  continue;
                default:
                  BUG();
              }
              return result;
            }
        }
      }
    }
    else if ( v13 == 11 )
    {
      if ( *(_BYTE *)(*(_QWORD *)(a1 + 24 * (1LL - v12)) + 16LL) != 13 )
        return 0;
      a1 = *(_QWORD *)(a1 - 24LL * v12);
    }
    else
    {
      if ( v13 != 32 )
        return v9;
      if ( v12 != 1 )
      {
        v14 = 1;
        while ( *(_BYTE *)(*(_QWORD *)(a1 + 24 * (v14 - (unsigned __int64)v12)) + 16LL) == 13 )
        {
          if ( v12 == ++v14 )
            goto LABEL_33;
        }
        return 0;
      }
LABEL_33:
      a1 = *(_QWORD *)(a1 - 24LL * v12);
    }
  }
  v17 = 3LL * v12;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
  {
    v18 = *(_QWORD **)(a1 - 8);
    v19 = &v18[v17];
  }
  else
  {
    v19 = (_QWORD *)a1;
    v18 = (_QWORD *)(a1 - v17 * 8);
  }
  do
  {
    v9 = sub_1AC5CF0(*v18, a2, a3);
    if ( !(_BYTE)v9 )
      break;
    v18 += 3;
  }
  while ( v19 != v18 );
  return v9;
}
