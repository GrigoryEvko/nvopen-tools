// Function: sub_1F2B030
// Address: 0x1f2b030
//
__int64 __fastcall sub_1F2B030(__int64 a1, __int64 a2, _BYTE *a3, unsigned __int8 a4, char a5)
{
  char v5; // al
  __int64 v6; // rbx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  unsigned int v12; // eax
  __int64 v13; // rcx
  unsigned __int64 v14; // r15
  __int64 v15; // rax
  _QWORD *v16; // r15
  unsigned int v17; // r14d
  unsigned int v18; // eax
  int v20; // eax
  __int64 v21; // rax
  _QWORD *v22; // rax
  unsigned int v23; // eax
  __int64 v24; // rsi
  __int64 v25; // rdx
  unsigned __int64 v26; // r10
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h]
  __int64 v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+28h] [rbp-48h]
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+28h] [rbp-48h]
  unsigned __int64 v37; // [rsp+38h] [rbp-38h]
  _QWORD *v38; // [rsp+38h] [rbp-38h]

  if ( !a2 )
    return 0;
  v5 = *(_BYTE *)(a2 + 8);
  v6 = a2;
  if ( v5 != 14 )
  {
LABEL_14:
    if ( v5 == 13 )
    {
      v16 = *(_QWORD **)(v6 + 16);
      v38 = &v16[*(unsigned int *)(v6 + 12)];
      if ( v38 != v16 )
      {
        v17 = 0;
        while ( 1 )
        {
          v18 = sub_1F2B030(a1, *v16, a3, a4, 1);
          if ( (_BYTE)v18 )
          {
            if ( *a3 )
              return 1;
            v17 = v18;
          }
          if ( ++v16 == v38 )
            return v17;
        }
      }
    }
    return 0;
  }
  if ( !sub_1642F90(*(_QWORD *)(a2 + 24), 8) && !a4 )
  {
    if ( a5 )
      return 0;
    v9 = *(unsigned int *)(a1 + 220);
    if ( (unsigned int)v9 > 0x1E )
      return 0;
    v10 = 1610614920;
    if ( !_bittest64(&v10, v9) )
      return 0;
  }
  v37 = *(unsigned int *)(a1 + 288);
  v11 = sub_1632FA0(*(_QWORD *)(a1 + 240));
  v12 = sub_15A9FE0(v11, a2);
  v13 = 1;
  v14 = v12;
  while ( 2 )
  {
    switch ( *(_BYTE *)(a2 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v21 = *(_QWORD *)(a2 + 32);
        a2 = *(_QWORD *)(a2 + 24);
        v13 *= v21;
        continue;
      case 1:
        v15 = 16;
        break;
      case 2:
        v15 = 32;
        break;
      case 3:
      case 9:
        v15 = 64;
        break;
      case 4:
        v15 = 80;
        break;
      case 5:
      case 6:
        v15 = 128;
        break;
      case 7:
        v32 = v13;
        v20 = sub_15A9520(v11, 0);
        v13 = v32;
        v15 = (unsigned int)(8 * v20);
        break;
      case 0xB:
        v15 = *(_DWORD *)(a2 + 8) >> 8;
        break;
      case 0xD:
        v33 = v13;
        v22 = (_QWORD *)sub_15A9930(v11, a2);
        v13 = v33;
        v15 = 8LL * *v22;
        break;
      case 0xE:
        v30 = v13;
        v31 = *(_QWORD *)(a2 + 24);
        v34 = *(_QWORD *)(a2 + 32);
        v23 = sub_15A9FE0(v11, v31);
        v13 = v30;
        v24 = v31;
        v25 = 1;
        v26 = v23;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v24 + 8) )
          {
            case 0:
              v29 = *(_QWORD *)(v24 + 32);
              v24 = *(_QWORD *)(v24 + 24);
              v25 *= v29;
              continue;
            case 1:
              v28 = 16;
              break;
            case 2:
              v28 = 32;
              break;
            case 3:
              v28 = 64;
              break;
          }
          break;
        }
        v15 = 8 * v34 * v26 * ((v26 + ((unsigned __int64)(v28 * v25 + 7) >> 3) - 1) / v26);
        break;
      case 0xF:
        v35 = v13;
        v27 = sub_15A9520(v11, *(_DWORD *)(a2 + 8) >> 8);
        v13 = v35;
        v15 = (unsigned int)(8 * v27);
        break;
    }
    break;
  }
  if ( v37 <= v14 * ((v14 + ((unsigned __int64)(v15 * v13 + 7) >> 3) - 1) / v14) )
  {
    *a3 = 1;
    return 1;
  }
  if ( !a4 )
  {
    v5 = *(_BYTE *)(v6 + 8);
    goto LABEL_14;
  }
  return 1;
}
