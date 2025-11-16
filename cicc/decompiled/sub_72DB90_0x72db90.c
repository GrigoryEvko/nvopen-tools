// Function: sub_72DB90
// Address: 0x72db90
//
__int64 __fastcall sub_72DB90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // eax
  unsigned int v10; // r12d
  __int64 v12; // rax
  int v13; // r12d
  int v14; // eax
  __int64 v15; // r12
  int v16; // eax
  int v17; // r12d
  int v18; // eax
  unsigned int v19; // eax
  __int64 v20; // rsi
  char *v21; // rax
  char *v22; // rsi
  int v23; // ecx
  int v24; // eax
  __int64 v25; // rdi
  int v26; // eax
  __int64 v27; // rdi
  int v28; // eax
  int v29; // eax
  int v30; // r12d
  __int64 v31; // rdi
  int v32; // eax
  __int64 v33; // rdi
  int v34; // eax
  __int64 v35; // r13
  int v36; // eax
  int v37; // eax
  __int64 v38; // rdi
  __int64 v39; // r13
  _QWORD *v40; // rax
  int v41; // eax
  int v42; // eax
  __int64 *v43; // rdi
  int v44; // eax
  int v45; // eax
  int v46; // eax
  int v47; // r12d
  int v48; // eax
  int v49; // eax
  __int64 v50; // rdi
  int v51; // r13d
  int v52; // eax
  int v53[9]; // [rsp+Ch] [rbp-24h] BYREF

  v7 = *(unsigned __int8 *)(a1 + 173);
  v8 = (unsigned __int8)v7;
  switch ( (char)v7 )
  {
    case 1:
      v19 = sub_620FA0(a1, v53);
      v7 = *(unsigned __int8 *)(a1 + 173);
      v10 = v19;
      goto LABEL_3;
    case 2:
      v20 = *(_QWORD *)(a1 + 176);
      v21 = *(char **)(a1 + 184);
      if ( v20 )
      {
        v22 = &v21[v20];
        v10 = 100;
        do
        {
          v23 = *v21++;
          v10 += v23 + 32 * v10;
        }
        while ( v22 != v21 );
      }
      else
      {
        return 100;
      }
      return v10;
    case 3:
    case 5:
      v9 = sub_70C920((unsigned __int8 *)(a1 + 176));
      v7 = *(unsigned __int8 *)(a1 + 173);
      v10 = v9 + 500;
      goto LABEL_3;
    case 4:
      v13 = sub_70C920(*(unsigned __int8 **)(a1 + 176));
      v14 = sub_70C920((unsigned __int8 *)(*(_QWORD *)(a1 + 176) + 16LL));
      v7 = *(unsigned __int8 *)(a1 + 173);
      v10 = v13 + v14 + 250;
      goto LABEL_3;
    case 6:
      switch ( *(_BYTE *)(a1 + 176) )
      {
        case 0:
          v35 = *(_QWORD *)(a1 + 184);
          v36 = *(_DWORD *)(v35 + 168);
          if ( v36 )
          {
            v10 = v36 + *(_QWORD *)(a1 + 192) + 1000;
            return (unsigned int)sub_72E3F0(*(_QWORD *)(a1 + 128)) + v10;
          }
          v37 = sub_72E220(*(_QWORD *)(a1 + 184));
          v38 = *(_QWORD *)(v35 + 240);
          v30 = v37;
          if ( v38 )
            v30 = sub_72E120(v38) + v37;
          if ( !v30 )
            v30 = 1;
          *(_DWORD *)(v35 + 168) = v30;
          v7 = *(unsigned __int8 *)(a1 + 173);
          goto LABEL_30;
        case 1:
          v39 = *(_QWORD *)(a1 + 184);
          v30 = sub_72E220(v39);
          v40 = *(_QWORD **)(v39 + 216);
          if ( !v40 )
            goto LABEL_43;
          v41 = sub_72E120(*v40);
          v7 = *(unsigned __int8 *)(a1 + 173);
          v30 += v41;
          goto LABEL_30;
        case 2:
          v33 = *(_QWORD *)(a1 + 184);
          if ( !*(_QWORD *)(v33 + 8) )
            goto LABEL_34;
          v30 = sub_72E220(v33);
LABEL_43:
          v7 = *(unsigned __int8 *)(a1 + 173);
          goto LABEL_30;
        case 3:
          v33 = *(_QWORD *)(a1 + 184);
LABEL_34:
          v34 = sub_72DB90(v33);
          v7 = *(unsigned __int8 *)(a1 + 173);
          v30 = v34;
          goto LABEL_30;
        case 5:
          v31 = *(_QWORD *)(a1 + 184);
          if ( !v31 )
          {
            v10 = *(_DWORD *)(a1 + 192) + 1233;
            return (unsigned int)sub_72E3F0(*(_QWORD *)(a1 + 128)) + v10;
          }
          v32 = sub_72E3F0(v31);
          v7 = *(unsigned __int8 *)(a1 + 173);
          v30 = v32 + 233;
LABEL_30:
          v10 = v30 + *(_QWORD *)(a1 + 192) + 1000;
          break;
        case 6:
          v29 = sub_72E220(*(_QWORD *)(a1 + 184));
          v7 = *(unsigned __int8 *)(a1 + 173);
          v30 = v29;
          goto LABEL_30;
        default:
          goto LABEL_55;
      }
      goto LABEL_3;
    case 7:
      v15 = *(_QWORD *)(a1 + 200);
      if ( (*(_BYTE *)(a1 + 192) & 2) != 0 )
      {
        if ( v15 )
        {
          v16 = *(_DWORD *)(v15 + 168);
          if ( v16 )
          {
            v10 = v16 + 250;
            return (unsigned int)sub_72E3F0(*(_QWORD *)(a1 + 128)) + v10;
          }
          v49 = sub_72E220(*(_QWORD *)(a1 + 200));
          v50 = *(_QWORD *)(v15 + 240);
          v51 = v49;
          if ( v50 )
            v51 = sub_72E120(v50) + v49;
          v52 = 1;
          if ( v51 )
            v52 = v51;
          *(_DWORD *)(v15 + 168) = v52;
          v10 = v52 + 250;
          v7 = *(unsigned __int8 *)(a1 + 173);
          goto LABEL_3;
        }
      }
      else if ( v15 )
      {
        v24 = sub_72E220(*(_QWORD *)(a1 + 200));
        v7 = *(unsigned __int8 *)(a1 + 173);
        v10 = v24 + 250;
        goto LABEL_3;
      }
      v10 = 250;
      return (unsigned int)sub_72E3F0(*(_QWORD *)(a1 + 128)) + v10;
    case 8:
      v17 = sub_72DB90(*(_QWORD *)(a1 + 176));
      v18 = sub_72DB90(*(_QWORD *)(a1 + 184));
      v7 = *(unsigned __int8 *)(a1 + 173);
      v10 = v17 + 3 * v18 + 511;
      goto LABEL_3;
    case 12:
      switch ( *(_BYTE *)(a1 + 176) )
      {
        case 0:
          v10 = a1 + 499;
          return (unsigned int)sub_72E3F0(*(_QWORD *)(a1 + 128)) + v10;
        case 1:
          if ( (*(_BYTE *)(a1 + 177) & 0x10) != 0 )
          {
            a2 = 3;
            v43 = sub_72DB50(a1, 3);
          }
          else
          {
            v43 = *(__int64 **)(a1 + 184);
          }
          v44 = sub_72A8B0((__int64)v43, a2, v7, v8, a5, a6);
          v7 = *(unsigned __int8 *)(a1 + 173);
          v10 = v44 + 499;
          goto LABEL_3;
        case 2:
          v42 = ((__int64 (*)(void))sub_72E220)();
          v7 = *(unsigned __int8 *)(a1 + 173);
          v10 = v42 + 499;
          goto LABEL_3;
        case 3:
          if ( *(_QWORD *)(a1 + 184) )
          {
            v46 = ((__int64 (*)(void))sub_72E3F0)();
            v7 = *(unsigned __int8 *)(a1 + 173);
            v10 = *(unsigned __int8 *)(a1 + 200) + v46 + 499;
            goto LABEL_3;
          }
          v10 = *(unsigned __int8 *)(a1 + 200) + 499;
          return (unsigned int)sub_72E3F0(*(_QWORD *)(a1 + 128)) + v10;
        case 4:
        case 0xC:
          v10 = sub_72DB90(*(_QWORD *)(a1 + 184)) + 499;
          goto LABEL_28;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 0xA:
          v25 = *(_QWORD *)(a1 + 184);
          if ( !v25 )
          {
            v27 = *(_QWORD *)(a1 + 192);
            v10 = 499;
            if ( !v27 )
              return (unsigned int)sub_72E3F0(*(_QWORD *)(a1 + 128)) + v10;
LABEL_25:
            v28 = sub_72A8B0(v27, a2, v7, v8, a5, a6);
            v7 = *(unsigned __int8 *)(a1 + 173);
            v10 += v28;
            goto LABEL_3;
          }
          v26 = sub_72E3F0(v25);
          v27 = *(_QWORD *)(a1 + 192);
          v10 = v26 + 499;
          if ( v27 )
            goto LABEL_25;
LABEL_28:
          v7 = *(unsigned __int8 *)(a1 + 173);
LABEL_3:
          if ( (unsigned __int8)v7 > 0xCu )
            return v10;
LABEL_6:
          v12 = 4290;
          if ( !_bittest64(&v12, v7) )
            return v10;
          return (unsigned int)sub_72E3F0(*(_QWORD *)(a1 + 128)) + v10;
        case 0xB:
          v47 = sub_72DB90(*(_QWORD *)(a1 + 184));
          v48 = sub_72E120(*(_QWORD *)(a1 + 192));
          v7 = *(unsigned __int8 *)(a1 + 173);
          v10 = v47 + v48 + 499;
          goto LABEL_3;
        case 0xD:
          v45 = sub_72E3F0(*(_QWORD *)(a1 + 184));
          v7 = *(unsigned __int8 *)(a1 + 173);
          v10 = v45 + 2 * (*(_BYTE *)(a1 + 192) & 1) + 499;
          goto LABEL_3;
        default:
LABEL_55:
          sub_721090();
      }
    default:
      v10 = (unsigned __int8)v7 + 200;
      if ( (unsigned __int8)v7 <= 0xCu )
        goto LABEL_6;
      return v10;
  }
}
